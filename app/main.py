"""FastAPI application for Stremio Arabic Subtitles Addon (Vercel Ready & Stateless)."""

import base64
import hashlib
import json
import logging
import re
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.cache import cache_manager
from app.config import settings
from app.extractor import (
    SubtitleExtractionError,
    extract_srt_from_zip,
    is_ass_subtitle,
    is_vtt_subtitle,
    transcode_to_utf8,
)
from app.models import Manifest, SubtitleItem, SubtitlesResponse, UserPreferences
from app.providers import (
    CinemetaClient,
    OpenSubtitlesProvider,
    SubdlProvider,
    SubsourceProvider,
    SubtitlecatProvider,
    YifysubtitlesProvider,
)
from app.services.aggregator import (
    aggregate_subtitles,
    format_informative_badge,
)
from app.services.cache import clear_subtitle_cache
from app.utils.ass_converter import convert_ass_to_srt_bytes
from app.utils.cleaners import (
    CleanOptions,
    clean_subtitle_bytes,
    convert_eastern_arabic_numerals_bytes,
    fix_subtitle_encoding_bytes,
    strip_advertisements_bytes,
    strip_arabic_diacritics_bytes,
    strip_hi_artifacts_bytes,
)
from app.utils.config_parser import parse_user_config
from app.utils.language import AVAILABLE_LANGUAGES, get_language_name, normalize_to_iso639_2
from app.utils.network import get_base_url, get_local_lan_ip, is_local_or_container_host
from app.utils.parser import parse_stremio_id
from app.utils.release_matcher import (
    extract_stream_params,
)
from app.utils.rtl import fix_rtl_punctuation_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stremio_arabic_subs")

_http_client: httpx.AsyncClient | None = None


async def get_http_client() -> httpx.AsyncClient:
    """Get or dynamically initialize HTTP client (Serverless friendly)."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=settings.MAX_KEEP_ALIVE_CONNECTIONS,
                max_connections=settings.MAX_CONNECTIONS,
                keepalive_expiry=30.0,
            ),
            timeout=httpx.Timeout(settings.UPSTREAM_TIMEOUT),
            headers={"User-Agent": "StremioArabicSubs/1.0.0"},
            follow_redirects=True,
        )
    return _http_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application resources, async HTTP connection pool, and graceful shutdown."""
    clear_subtitle_cache()
    cache_manager.clear_metadata()
    logger.info("Cleared in-memory subtitle TTLCache and metadata cache on startup.")
    await get_http_client()
    yield
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


app = FastAPI(
    title="NinjaSubs",
    version="1.0.0",
    description="Smart, high-accuracy subtitle aggregator for Stremio featuring advanced Arabic subtitle optimization.",
    lifespan=lifespan,
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    """Serve official addon icon for browser favicon requests."""
    icon_file = static_dir / "icon.png"
    if icon_file.is_file():
        return FileResponse(icon_file, media_type="image/png")
    return Response(status_code=404)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_global_cors_headers(request: Request, call_next):
    """Ensure CORS headers are always present on all responses."""
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


def _build_manifest(config_str: str | None = None, request: Request | None = None) -> Manifest:
    """Build Stremio Manifest object with community-standard behaviorHints."""
    desc = "Smart, high-accuracy subtitle aggregator for Stremio featuring advanced Arabic subtitle optimization."
    base_url = get_base_url(request)
    icon_url = f"{base_url}/static/icon.png"
    logo_url = f"{base_url}/static/icon.png"

    return Manifest(
        id="org.ninjasubs.addon",
        name="NinjaSubs",
        version="1.0.0",
        description=desc,
        logo=logo_url,
        icon=icon_url,
        resources=["subtitles"],
        types=["movie", "series", "anime"],
        idPrefixes=["tt", "kitsu"],
        catalogs=[],
        behaviorHints={
            "configurable": True,
            "configurationRequired": False,
        },
    )


def render_configure_html(request: Request, prefill_config: str | None = None) -> str:
    """Render configuration page."""
    lan_ip = get_local_lan_ip()
    port = request.url.port or settings.PORT or 7000
    lan_url = f"http://{lan_ip}:{port}"
    req_host = request.url.hostname or ""
    base_url = lan_url if is_local_or_container_host(req_host) else get_base_url(request)

    subdl_env_set = bool(settings.SUBDL_API_KEY.strip())
    subsource_env_set = bool(settings.SUBSOURCE_API_KEY.strip())
    opensubtitles_env_set = bool(settings.OPENSUBTITLES_API_KEY.strip())

    if prefill_config:
        prefs = parse_user_config(prefill_config)
    else:
        prefs = UserPreferences()

    initial_subdl = prefs.subdl_key if prefs.subdl_key != settings.SUBDL_API_KEY else ""
    initial_subsource = (
        prefs.subsource_key if prefs.subsource_key != settings.SUBSOURCE_API_KEY else ""
    )
    initial_opensubtitles = (
        prefs.opensubtitles_key if prefs.opensubtitles_key != settings.OPENSUBTITLES_API_KEY else ""
    )
    initial_exclude_hi = "checked" if prefs.exclude_hi else ""

    _fresh_defaults = not prefill_config
    initial_phase2_json = json.dumps(
        {
            "badge_parts": prefs.resolved_badge_parts,
            "enable_subdl": prefs.enable_subdl,
            "enable_subsource": prefs.enable_subsource,
            "enable_opensubtitles": prefs.enable_opensubtitles,
            "enable_yifysubtitles": prefs.enable_yifysubtitles,
            "enable_subtitlecat": prefs.enable_subtitlecat,
            "enable_rtl_fix": prefs.enable_rtl_fix,
            "enable_ad_removal": prefs.enable_ad_removal,
            "keep_translator_credits": prefs.keep_translator_credits,
            "fix_encoding": prefs.fix_encoding,
            "clean_tags": prefs.clean_tags,
            "strip_colors": prefs.strip_colors,
            "clean_spacing": False if _fresh_defaults else prefs.clean_spacing,
            "clean_symbols": False if _fresh_defaults else prefs.clean_symbols,
            "clean_commas": False if _fresh_defaults else prefs.clean_commas,
            "clean_timing": False if _fresh_defaults else prefs.clean_timing,
            "strip_hi": prefs.strip_hi,
            "eastern_arabic_numerals": prefs.eastern_arabic_numerals,
            "strip_diacritics": prefs.strip_diacritics,
            "convert_ass_to_srt": prefs.convert_ass_to_srt,
        }
    )

    if prefill_config:
        pref_langs = [normalize_to_iso639_2(lang) for lang in prefs.languages] if prefs.languages else ["ara"]
    else:
        pref_langs = []
    options_html = []
    seen_codes = set()
    for lang in AVAILABLE_LANGUAGES:
        code = lang["code"]
        name = lang["name"]
        seen_codes.add(code)
        selected = "selected" if code in pref_langs else ""
        options_html.append(f'<option value="{code}" {selected}>{name} ({code})</option>')
    for p_code in pref_langs:
        if p_code not in seen_codes:
            options_html.append(
                f'<option value="{p_code}" selected>{p_code.upper()} ({p_code})</option>'
            )
    language_options_html = "\n                    ".join(options_html)

    env_banner_html = ""
    if subdl_env_set or subsource_env_set or opensubtitles_env_set:
        env_banner_html = f"""
        <div class="flex items-center justify-between gap-3 px-4 py-3 rounded-2xl bg-[#141619] border border-[#22262b] text-xs">
            <div class="flex items-center gap-2">
                <span class="h-2 w-2 rounded-full bg-emerald-400"></span>
                <span class="font-medium text-white">Server Defaults Active</span>
            </div>
            <div class="text-[#8b929a] font-mono text-[11px]">
                Subdl: {'&#10004; Configured' if subdl_env_set else '&#9888; Not Set'} &bull;
                Subsource: {'&#10004; Configured' if subsource_env_set else '&#9888; Not Set'} &bull;
                OpenSubtitles: {'&#10004; Configured' if opensubtitles_env_set else '&#9888; Not Set'}
            </div>
        </div>
        """

    template_path = Path(__file__).parent / "templates" / "configure.html"
    if template_path.is_file():
        raw_html = template_path.read_text(encoding="utf-8")
        return (
            raw_html.replace("{{base_url}}", base_url)
            .replace("{{lan_ip}}", lan_ip)
            .replace("{{lan_url}}", lan_url)
            .replace("{{port}}", str(port))
            .replace("{{initial_subdl}}", initial_subdl)
            .replace("{{initial_subsource}}", initial_subsource)
            .replace("{{initial_opensubtitles}}", initial_opensubtitles)
            .replace("{{initial_exclude_hi}}", initial_exclude_hi)
            .replace("{{initial_phase2_json}}", initial_phase2_json)
            .replace("{{language_options_html}}", language_options_html)
            .replace("{{env_banner_html}}", env_banner_html)
        )

    return "<html><body>Configure page template missing</body></html>"


@app.get("/api/verify/subdl")
async def verify_subdl_endpoint(api_key: str | None = None):
    if not api_key or not api_key.strip():
        return {"valid": False, "message": "API key is required"}

    key = api_key.strip()
    client = await get_http_client()

    try:
        resp = await client.get(
            "https://api.subdl.com/api/v1/subtitles",
            params={"api_key": key, "imdb_id": "tt0111161", "type": "movie"},
            headers={"Accept": "application/json", "User-Agent": "StremioArabicSubs/1.0.0"},
            timeout=settings.UPSTREAM_TIMEOUT,
        )
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {}
            if data.get("status") is False:
                status_code = data.get("statusCode")
                err_text = str(data.get("error", "")).lower()
                if status_code in (401, 403) or "not_authorized" in err_text or "invalid" in err_text:
                    return {"valid": False, "message": data.get("message", "Invalid API key")}
            return {"valid": True, "message": "Valid API key"}
        elif resp.status_code in (401, 403):
            return {"valid": False, "message": "Invalid API key"}
        elif resp.status_code == 429:
            return {"valid": True, "message": "Valid API key (rate limited)"}
        else:
            return {"valid": False, "message": f"Unexpected response (HTTP {resp.status_code})"}
    except Exception as e:
        logger.warning(f"Subdl verification connection error: {e}")
        return {"valid": False, "error": "Connection error", "message": str(e)}


uvicorn_logger = logging.getLogger("uvicorn.error")


@app.get("/api/verify/subsource")
async def verify_subsource_endpoint(api_key: str | None = None):
    if not api_key or not api_key.strip():
        return {"valid": False, "message": "API key is required"}

    key = api_key.strip()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "X-API-Key": key,
        "Authorization": f"Bearer {key}",
    }

    test_urls = [
        ("https://api.subsource.net/api/v1/subtitles", {"imdb_id": "tt0903747"}),
        (f"https://api.subsource.net/api/v1/subtitles?apiKey={key}&imdb_id=tt0903747", None),
        (f"https://api.subsource.net/api/v1/subtitles?api_key={key}&imdb_id=tt0903747", None),
        ("https://api.subsource.net/api/v1/user", None),
        ("https://api.subsource.net/api/v1/movies/search", {"searchType": "imdb", "q": "tt0903747"}),
    ]

    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
        last_status = None
        last_body = ""

        for url, params in test_urls:
            try:
                resp = await client.get(url, headers=headers, params=params)
                last_status = resp.status_code
                last_body = resp.text[:200]

                if resp.status_code in (200, 404):
                    return {"valid": True}
                if resp.status_code in (401, 403):
                    continue
            except Exception as e:
                uvicorn_logger.error(f"[SubSource Check Error] {str(e)}")
                last_body = str(e)

        return {
            "valid": False,
            "status_code": last_status,
            "error_snippet": last_body,
            "message": "Invalid API Key or unauthorized",
        }


@app.get("/api/verify/opensubtitles")
async def verify_opensubtitles_key(api_key: str | None = None):
    if not api_key or not api_key.strip():
        return {"valid": False, "message": "API key is required"}

    headers = {
        "Api-Key": api_key.strip(),
        "User-Agent": "StremioArabicSubs v1.0.0",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
        try:
            resp = await client.get(
                "https://api.opensubtitles.com/api/v1/subtitles?imdb_id=111161&languages=en",
                headers=headers,
            )
            if resp.status_code == 200:
                return {"valid": True}
            return {"valid": False, "status": resp.status_code, "detail": resp.text[:100]}
        except Exception as e:
            return {"valid": False, "error": "Connection error", "message": str(e)}


@app.get("/", response_class=HTMLResponse)
@app.get("/configure", response_class=HTMLResponse)
async def configure_page(request: Request):
    return HTMLResponse(content=render_configure_html(request))


@app.get("/{config}/configure", response_class=HTMLResponse)
async def configure_prefill_page(config: str, request: Request):
    return HTMLResponse(content=render_configure_html(request, prefill_config=config))


@app.api_route("/manifest.json", methods=["GET", "HEAD", "OPTIONS"], response_model=Manifest)
@app.api_route("/manifest", methods=["GET", "HEAD", "OPTIONS"], response_model=Manifest)
async def get_manifest(request: Request):
    return _build_manifest(request=request)


@app.api_route(
    "/{config}/manifest.json", methods=["GET", "HEAD", "OPTIONS"], response_model=Manifest
)
@app.api_route("/{config}/manifest", methods=["GET", "HEAD", "OPTIONS"], response_model=Manifest)
async def get_configured_manifest(config: str, request: Request):
    return _build_manifest(config_str=config, request=request)


async def _fetch_subtitles_handler(
    media_type: str,
    raw_id: str,
    request: Request,
    config_str: str | None = None,
    extra: str | None = None,
) -> SubtitlesResponse:
    client = await get_http_client()

    try:
        parsed = parse_stremio_id(raw_id)
    except ValueError as e:
        logger.warning(f"Failed to parse Stremio ID '{raw_id}': {e}")
        return SubtitlesResponse(subtitles=[])

    try:
        query_subdl = request.query_params.get("subdl_key") or request.query_params.get("subdl_api_key")
        query_subsource = request.query_params.get("subsource_key") or request.query_params.get("subsource_api_key")
        query_opensubtitles = request.query_params.get("opensubtitles_key") or request.query_params.get("opensubtitles_api_key")
        prefs = parse_user_config(
            config_str,
            query_subdl=query_subdl,
            query_subsource=query_subsource,
            query_opensubtitles=query_opensubtitles,
        )
    except Exception:
        prefs = UserPreferences()

    subdl = SubdlProvider(client)
    subsource = SubsourceProvider(client)
    opensubtitles = OpenSubtitlesProvider(client)
    cinemeta = CinemetaClient(client)

    is_series = parsed.is_series
    season = parsed.season
    episode = parsed.episode

    if media_type.lower() == "anime":
        if episode is not None:
            is_series = True
            if season is None:
                season = 1
        elif is_series and season is None:
            season = 1

    cinemeta_type = "series" if is_series else ("movie" if media_type.lower() == "movie" else "series")
    meta_info = await cinemeta.get_metadata(cinemeta_type, parsed.imdb_id)
    title = meta_info.get("title") if meta_info else None
    year = meta_info.get("year") if meta_info else None

    stream_params = extract_stream_params(extra, request.query_params)
    target_filename = stream_params.get("filename")
    video_hash = stream_params.get("video_hash")
    video_size = stream_params.get("video_size")

    bypass_cache = bool(
        request.query_params.get("nocache")
        or request.query_params.get("refresh")
        or request.query_params.get("bypass_cache")
        or request.headers.get("x-bypass-cache")
    )

    ranked_releases = await aggregate_subtitles(
        imdb_id=parsed.imdb_id,
        media_type=media_type,
        season=season,
        episode=episode,
        filename=target_filename,
        video_hash=video_hash,
        video_size=video_size,
        languages=prefs.languages,
        exclude_hi=prefs.exclude_hi,
        user_preferences=prefs,
        title=title,
        year=year,
        http_client=client,
        subdl_provider=subdl,
        subsource_provider=subsource,
        opensubtitles_provider=opensubtitles,
        use_cache=not bypass_cache,
    )

    base_url = get_base_url(request)
    subtitle_items: list[SubtitleItem] = []

    for rel in ranked_releases:
        display_score = getattr(rel, "match_percentage", None)
        if display_score is None:
            display_score = rel.score

        unique_key = f"{rel.provider}:{rel.release_name}:{rel.download_url}"
        sub_id = hashlib.sha256(unique_key.encode("utf-8")).hexdigest()[:16]

        rel_lang = normalize_to_iso639_2(getattr(rel, "lang", "ara") or "ara")
        rel_prov = (rel.provider or "").strip().lower()
        if rel_prov == "subsource":
            source_tag = "SubSource"
        elif rel_prov == "opensubtitles":
            source_tag = "OpenSubtitles"
        elif rel_prov == "yifysubtitles":
            source_tag = "YIFY"
        elif rel_prov == "subtitlecat":
            source_tag = "SubtitleCat"
        else:
            source_tag = "SubDL"

        meta_dict = {
            "sub_id": sub_id,
            "imdb_id": parsed.imdb_id,
            "media_type": media_type,
            "provider": rel.provider,
            "download_url": rel.download_url,
            "release_name": rel.release_name,
            "target_filename": target_filename or rel.release_name,
            "season": season,
            "episode": episode,
            "subdl_key": prefs.subdl_key,
            "subsource_key": prefs.subsource_key,
            "opensubtitles_key": prefs.opensubtitles_key,
            "lang": rel_lang,
        }
        cache_manager.store_metadata(sub_id, meta_dict)

        lang_name = get_language_name(rel_lang, default="Arabic")
        display_label = format_informative_badge(
            rel,
            display_score,
            lang_name=lang_name,
            source_tag=source_tag,
            badge_parts=prefs.resolved_badge_parts,
        )

        track_lang = rel_lang
        track_id = display_label

        sub_format = getattr(rel, "format", "srt") or "srt"
        r_name_lower = rel.release_name.lower()
        if r_name_lower.endswith(".ass"):
            sub_format = "ass"
        elif r_name_lower.endswith(".ssa"):
            sub_format = "ssa"
        elif r_name_lower.endswith(".vtt"):
            sub_format = "vtt"

        # Stateful & Stateless Payload encoding for serverless environments
        payload = {
            "p": rel.provider,
            "u": rel.download_url,
            "r": rel.release_name,
            "s": season,
            "e": episode,
            "lang": rel_lang,
            "id": parsed.imdb_id,
        }
        encoded_token = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")

        if rel_prov == "opensubtitles":
            m_fid = re.search(r"(\d+)", rel.download_url)
            file_id = m_fid.group(1) if m_fid else encoded_token
            cache_manager.store_metadata(str(file_id), meta_dict)
            sub_url = (
                f"{base_url}/{config_str}/sub/opensubtitles/{file_id}.{sub_format}"
                if config_str
                else f"{base_url}/sub/opensubtitles/{file_id}.{sub_format}"
            )
        else:
            sub_url = (
                f"{base_url}/{config_str}/sub/{encoded_token}.{sub_format}"
                if config_str
                else f"{base_url}/sub/{encoded_token}.{sub_format}"
            )

        subtitle_items.append(
            SubtitleItem(
                id=track_id,
                url=sub_url,
                lang=track_lang,
                title=display_label,
                format=sub_format,
            )
        )

    seen_track_keys: set[tuple[str, str]] = set()
    unique_items: list[SubtitleItem] = []
    for item in subtitle_items:
        dedup_key = (item.id, item.lang)
        if dedup_key in seen_track_keys:
            continue
        seen_track_keys.add(dedup_key)
        unique_items.append(item)

    return SubtitlesResponse(subtitles=unique_items)


@app.api_route("/subtitles/{media_type}/{media_id}.json", methods=["GET", "HEAD", "OPTIONS"], response_model=SubtitlesResponse)
@app.api_route("/subtitles/{media_type}/{media_id}", methods=["GET", "HEAD", "OPTIONS"], response_model=SubtitlesResponse)
async def get_subtitles(media_type: str, media_id: str, request: Request):
    return await _fetch_subtitles_handler(media_type, media_id, request)


@app.api_route("/subtitles/{media_type}/{media_id}/{extra:path}.json", methods=["GET", "HEAD", "OPTIONS"], response_model=SubtitlesResponse)
@app.api_route("/subtitles/{media_type}/{media_id}/{extra:path}", methods=["GET", "HEAD", "OPTIONS"], response_model=SubtitlesResponse)
async def get_subtitles_with_extra(media_type: str, media_id: str, extra: str, request: Request):
    return await _fetch_subtitles_handler(media_type, media_id, request, extra=extra)


@app.api_route("/{config}/subtitles/{media_type}/{media_id}.json", methods=["GET", "HEAD", "OPTIONS"], response_model=SubtitlesResponse)
@app.api_route("/{config}/subtitles/{media_type}/{media_id}", methods=["GET", "HEAD", "OPTIONS"], response_model=SubtitlesResponse)
async def get_configured_subtitles(config: str, media_type: str, media_id: str, request: Request):
    return await _fetch_subtitles_handler(media_type, media_id, request, config_str=config)


@app.api_route("/{config}/subtitles/{media_type}/{media_id}/{extra:path}.json", methods=["GET", "HEAD", "OPTIONS"], response_model=SubtitlesResponse)
@app.api_route("/{config}/subtitles/{media_type}/{media_id}/{extra:path}", methods=["GET", "HEAD", "OPTIONS"], response_model=SubtitlesResponse)
async def get_configured_subtitles_with_extra(config: str, media_type: str, media_id: str, extra: str, request: Request):
    return await _fetch_subtitles_handler(media_type, media_id, request, config_str=config, extra=extra)


def srt_to_vtt(srt_bytes: bytes) -> bytes:
    try:
        srt_text = srt_bytes.decode("utf-8", errors="replace")
        lines = srt_text.replace("\r\n", "\n").splitlines()
        vtt_lines = ["WEBVTT\n"]
        for line in lines:
            if " --> " in line:
                line = re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", line)
            vtt_lines.append(line)
        return "\n".join(vtt_lines).encode("utf-8")
    except Exception:
        return srt_bytes


async def _fallback_download_subsource(
    imdb_id: str,
    media_type: str = "series",
    season: int | None = None,
    episode: int | None = None,
    subsource_key: str | None = None,
    target_filename: str | None = None,
    lang: str = "ara",
    client: httpx.AsyncClient | None = None,
) -> bytes | None:
    effective_key = (subsource_key or getattr(settings, "SUBSOURCE_API_KEY", "") or "").strip()
    if not effective_key:
        return None

    if client is None:
        client = await get_http_client()

    try:
        from app.services.subtitle_matcher import rank_subtitles

        provider = SubsourceProvider(client)
        is_series = (str(media_type).lower() in ("series", "tv", "anime")) or (season is not None and episode is not None)
        releases = await provider.search_subtitles(
            imdb_id=imdb_id,
            is_series=is_series,
            season=season,
            episode=episode,
            api_key=effective_key,
            languages=[lang],
            target_filename=target_filename,
        )
        if not releases:
            return None

        if target_filename or episode is not None:
            ranked = rank_subtitles(
                video_filename=target_filename,
                subtitles=releases,
                preferred_languages=[lang],
                discard_mismatches=False,
                season=season,
                episode=episode,
            )
            candidates = ranked if ranked else releases
        else:
            candidates = releases

        best_release = candidates[0]
        raw_data = await provider.download_archive(best_release.download_url, api_key=effective_key)
        if not raw_data:
            return None

        if raw_data[:4] == b"PK\x03\x04":
            srt_bytes = extract_srt_from_zip(
                raw_data,
                target_filename=target_filename or best_release.release_name,
                season=season,
                episode=episode,
            )
        else:
            srt_bytes = transcode_to_utf8(raw_data)

        return srt_bytes
    except Exception as e:
        logger.error(f"[Subsource Fallback] Exception: {e}")
        return None


def _format_content_disposition(filename: Any, ext: str) -> str:
    base = re.sub(r"\.(?:srt|vtt|ass|ssa|sub)$", "", str(filename).strip(), flags=re.IGNORECASE)
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    clean_ext = ext.lstrip(".")
    return f'inline; filename="{safe}.{clean_ext}"'


_CLEAN_OPTION_DEFAULTS: dict[str, bool] = {
    "fix_encoding": True,
    "clean_tags": True,
    "strip_colors": False,
    "clean_spacing": True,
    "clean_symbols": True,
    "clean_commas": True,
    "clean_timing": True,
}


def _clean_options_from_query(query_params: Any) -> CleanOptions:
    values = dict(_CLEAN_OPTION_DEFAULTS)
    for name, default in _CLEAN_OPTION_DEFAULTS.items():
        if name in query_params:
            raw = query_params.get(name, "1" if default else "0")
            values[name] = str(raw).lower() not in ("0", "false", "no")
    return CleanOptions(**values)


def _run_subtitle_optimization_pipeline(
    sub_bytes: bytes,
    *,
    enable_rtl_fix: bool = True,
    enable_ad_removal: bool = True,
    keep_translator_credits: bool = True,
    options: CleanOptions | None = None,
    strip_hi: bool = False,
    eastern_arabic_numerals: bool = False,
    strip_diacritics: bool = False,
) -> bytes:
    options = options or CleanOptions()
    if strip_hi:
        sub_bytes = strip_hi_artifacts_bytes(sub_bytes)
    if enable_ad_removal:
        sub_bytes = strip_advertisements_bytes(sub_bytes, keep_translator_credits)
    if strip_diacritics:
        sub_bytes = strip_arabic_diacritics_bytes(sub_bytes)
    sub_bytes = clean_subtitle_bytes(sub_bytes, options)
    if enable_rtl_fix:
        sub_bytes = fix_rtl_punctuation_bytes(sub_bytes)
    if eastern_arabic_numerals:
        sub_bytes = convert_eastern_arabic_numerals_bytes(sub_bytes)
    return sub_bytes


def _build_subtitle_response(
    sub_bytes: bytes,
    release_name: Any,
    req_format: str = "srt",
    enable_rtl_fix: bool = True,
    enable_ad_removal: bool = True,
    keep_translator_credits: bool = True,
    clean_options: CleanOptions | None = None,
    strip_hi: bool = False,
    eastern_arabic_numerals: bool = False,
    strip_diacritics: bool = False,
    convert_ass: bool = True,
) -> Response:
    options = clean_options or CleanOptions()
    if options.fix_encoding:
        sub_bytes = fix_subtitle_encoding_bytes(sub_bytes)

    converted_from_ass = False
    if convert_ass and (is_ass_subtitle(sub_bytes) or req_format in ("ass", "ssa")):
        sub_bytes = convert_ass_to_srt_bytes(sub_bytes, apply_rtl=False)
        req_format = "srt"
        converted_from_ass = True

    sub_bytes = _run_subtitle_optimization_pipeline(
        sub_bytes,
        enable_rtl_fix=enable_rtl_fix,
        enable_ad_removal=enable_ad_removal,
        keep_translator_credits=keep_translator_credits,
        options=options,
        strip_hi=strip_hi,
        eastern_arabic_numerals=eastern_arabic_numerals,
        strip_diacritics=strip_diacritics,
    )

    if (not converted_from_ass) and (is_ass_subtitle(sub_bytes) or req_format in ("ass", "ssa")):
        ext = "ssa" if req_format == "ssa" else "ass"
        return Response(
            content=sub_bytes,
            media_type="text/x-ssa; charset=utf-8",
            headers={
                "Cache-Control": "public, max-age=86400",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Content-Disposition": _format_content_disposition(release_name, ext),
            },
        )
    elif is_vtt_subtitle(sub_bytes) or req_format == "vtt":
        vtt_bytes = sub_bytes if is_vtt_subtitle(sub_bytes) else srt_to_vtt(sub_bytes)
        return Response(
            content=vtt_bytes,
            media_type="text/vtt; charset=utf-8",
            headers={
                "Cache-Control": "public, max-age=86400",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Content-Disposition": _format_content_disposition(release_name, "vtt"),
            },
        )
    else:
        return Response(
            content=sub_bytes,
            media_type="application/x-subrip; charset=utf-8",
            headers={
                "Cache-Control": "public, max-age=86400",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Content-Disposition": _format_content_disposition(release_name, "srt"),
            },
        )


async def _serve_subtitle_handler(
    sub_id: str,
    config_str: str | None = None,
    req_format: str | None = None,
    is_vtt: bool = False,
) -> Response:
    client = await get_http_client()

    rtl_fix_enabled = True
    ad_removal_enabled = True
    keep_credits_enabled = True
    clean_options = CleanOptions()
    strip_hi_enabled = False
    eastern_numerals_enabled = False
    strip_diacritics_enabled = False
    convert_ass_enabled = True
    cfg_prefs = UserPreferences()
    if config_str:
        try:
            cfg_prefs = parse_user_config(config_str)
            rtl_fix_enabled = cfg_prefs.enable_rtl_fix
            ad_removal_enabled = cfg_prefs.enable_ad_removal
            keep_credits_enabled = cfg_prefs.keep_translator_credits
            clean_options = CleanOptions.from_prefs(cfg_prefs)
            strip_hi_enabled = cfg_prefs.strip_hi
            eastern_numerals_enabled = cfg_prefs.eastern_arabic_numerals
            strip_diacritics_enabled = cfg_prefs.strip_diacritics
            convert_ass_enabled = cfg_prefs.convert_ass_to_srt
        except Exception:
            pass

    clean_sub_id = urllib.parse.unquote(sub_id).strip()
    detected_format = req_format or ("vtt" if is_vtt else "srt")
    if clean_sub_id.endswith(".ass"):
        clean_sub_id = clean_sub_id[:-4]
        detected_format = "ass"
    elif clean_sub_id.endswith(".ssa"):
        clean_sub_id = clean_sub_id[:-4]
        detected_format = "ssa"
    elif clean_sub_id.endswith(".vtt"):
        clean_sub_id = clean_sub_id[:-4]
        detected_format = "vtt"
    elif clean_sub_id.endswith(".srt"):
        clean_sub_id = clean_sub_id[:-4]
        detected_format = "srt"

    target_id = clean_sub_id

    # 1. Check local cache
    cached_content = await cache_manager.get_subtitle(target_id)
    if cached_content:
        meta = cache_manager.get_metadata(target_id)
        release_name = meta.get("release_name", target_id) if meta else target_id
        return _build_subtitle_response(
            cached_content,
            release_name,
            detected_format,
            rtl_fix_enabled,
            ad_removal_enabled,
            keep_credits_enabled,
            clean_options,
            strip_hi_enabled,
            eastern_numerals_enabled,
            strip_diacritics_enabled,
            convert_ass_enabled,
        )

    # 2. Retrieve metadata: from memory or by decoding token payload directly
    meta = cache_manager.get_metadata(target_id)
    if not meta:
        try:
            padded = target_id + "=" * (-len(target_id) % 4)
            decoded_json = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
            payload = json.loads(decoded_json)
            meta = {
                "provider": payload.get("p"),
                "download_url": payload.get("u"),
                "release_name": payload.get("r", target_id),
                "season": payload.get("s"),
                "episode": payload.get("e"),
                "lang": payload.get("lang", "ara"),
                "imdb_id": payload.get("id"),
                "subdl_key": cfg_prefs.subdl_key,
                "subsource_key": cfg_prefs.subsource_key,
                "opensubtitles_key": cfg_prefs.opensubtitles_key,
            }
        except Exception:
            raise HTTPException(status_code=404, detail="Subtitle metadata not found or expired")

    provider_name = meta.get("provider")
    download_url = meta.get("download_url")
    release_name = meta.get("release_name", target_id)
    season = meta.get("season")
    episode = meta.get("episode")

    subdl_key = meta.get("subdl_key") or cfg_prefs.subdl_key
    subsource_key = meta.get("subsource_key") or cfg_prefs.subsource_key
    opensubtitles_key = meta.get("opensubtitles_key") or cfg_prefs.opensubtitles_key

    if not download_url:
        raise HTTPException(status_code=404, detail="Missing download URL for subtitle")

    if provider_name == "subdl":
        provider = SubdlProvider(client)
        raw_archive = await provider.download_archive(download_url, api_key=subdl_key)
    elif provider_name == "subsource":
        provider = SubsourceProvider(client)
        raw_archive = await provider.download_archive(download_url, api_key=subsource_key)
    elif provider_name == "opensubtitles":
        provider = OpenSubtitlesProvider(client)
        raw_archive = await provider.download_archive(download_url, api_key=opensubtitles_key)
    elif provider_name == "yifysubtitles":
        provider = YifysubtitlesProvider(client)
        raw_archive = await provider.download_archive(download_url)
    elif provider_name == "subtitlecat":
        provider = SubtitlecatProvider(client)
        raw_archive = await provider.download_archive(download_url)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider_name}")

    if not raw_archive:
        if meta.get("imdb_id") and provider_name != "subsource":
            fallback_bytes = await _fallback_download_subsource(
                imdb_id=meta["imdb_id"],
                media_type=meta.get("media_type") or ("series" if season is not None else "movie"),
                season=season,
                episode=episode,
                subsource_key=subsource_key,
                target_filename=meta.get("target_filename") or release_name,
                lang=meta.get("lang", "ara"),
                client=client,
            )
            if fallback_bytes:
                await cache_manager.save_subtitle(target_id, fallback_bytes)
                return _build_subtitle_response(
                    fallback_bytes,
                    release_name,
                    detected_format,
                    rtl_fix_enabled,
                    ad_removal_enabled,
                    keep_credits_enabled,
                    clean_options,
                    strip_hi_enabled,
                    eastern_numerals_enabled,
                    strip_diacritics_enabled,
                    convert_ass_enabled,
                )

        raise HTTPException(status_code=502, detail="Failed to download subtitle from upstream provider")

    try:
        if raw_archive[:4] == b"PK\x03\x04":
            srt_bytes = extract_srt_from_zip(
                raw_archive,
                target_filename=release_name,
                season=season,
                episode=episode,
            )
        else:
            srt_bytes = transcode_to_utf8(raw_archive)
    except Exception as e:
        logger.error(f"Error extracting subtitle #{target_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal subtitle extraction error") from e

    await cache_manager.save_subtitle(target_id, srt_bytes)

    return _build_subtitle_response(
        srt_bytes,
        release_name,
        detected_format,
        rtl_fix_enabled,
        ad_removal_enabled,
        keep_credits_enabled,
        clean_options,
        strip_hi_enabled,
        eastern_numerals_enabled,
        strip_diacritics_enabled,
        convert_ass_enabled,
    )


@app.api_route("/sub/opensubtitles/{file_id}.srt", methods=["GET", "HEAD"])
@app.api_route("/sub/opensubtitles/{file_id}.ass", methods=["GET", "HEAD"])
@app.api_route("/sub/opensubtitles/{file_id}.vtt", methods=["GET", "HEAD"])
@app.api_route("/{config}/sub/opensubtitles/{file_id}.srt", methods=["GET", "HEAD"])
@app.api_route("/{config}/sub/opensubtitles/{file_id}.ass", methods=["GET", "HEAD"])
@app.api_route("/{config}/sub/opensubtitles/{file_id}.vtt", methods=["GET", "HEAD"])
async def proxy_opensubtitles_stream(file_id: int, request: Request, config: str | None = None):
    path = request.url.path.lower()
    if path.endswith(".vtt"):
        req_fmt = "vtt"
    elif path.endswith(".ass") or path.endswith(".ssa"):
        req_fmt = "ass"
    else:
        req_fmt = "srt"

    rtl_fix_enabled = True
    ad_removal_enabled = True
    keep_credits_enabled = True
    clean_options = CleanOptions()
    strip_hi_enabled = False
    eastern_numerals_enabled = False
    strip_diacritics_enabled = False
    convert_ass_enabled = True
    if config:
        try:
            cfg_prefs = parse_user_config(config)
            rtl_fix_enabled = cfg_prefs.enable_rtl_fix
            ad_removal_enabled = cfg_prefs.enable_ad_removal
            keep_credits_enabled = cfg_prefs.keep_translator_credits
            clean_options = CleanOptions.from_prefs(cfg_prefs)
            strip_hi_enabled = cfg_prefs.strip_hi
            eastern_numerals_enabled = cfg_prefs.eastern_arabic_numerals
            strip_diacritics_enabled = cfg_prefs.strip_diacritics
            convert_ass_enabled = cfg_prefs.convert_ass_to_srt
        except Exception:
            pass

    cached_content = await cache_manager.get_subtitle(str(file_id))
    if cached_content:
        meta = cache_manager.get_metadata(str(file_id))
        release_name = meta.get("release_name", file_id) if meta else file_id
        return _build_subtitle_response(
            cached_content,
            release_name,
            req_fmt,
            rtl_fix_enabled,
            ad_removal_enabled,
            keep_credits_enabled,
            clean_options,
            strip_hi_enabled,
            eastern_numerals_enabled,
            strip_diacritics_enabled,
            convert_ass_enabled,
        )

    api_key = ""
    subsource_key = ""
    if config:
        cfg_prefs = parse_user_config(config)
        api_key = cfg_prefs.opensubtitles_key
        subsource_key = cfg_prefs.subsource_key
    if not api_key:
        api_key = request.query_params.get("opensubtitles_key") or getattr(settings, "OPENSUBTITLES_API_KEY", "") or ""
    if not subsource_key:
        subsource_key = request.query_params.get("subsource_key") or getattr(settings, "SUBSOURCE_API_KEY", "") or ""

    client = await get_http_client()
    opensubtitles_provider = OpenSubtitlesProvider(client)
    download_url = await opensubtitles_provider.get_download_url(file_id, api_key)

    if download_url:
        try:
            dl_resp = await client.get(download_url, follow_redirects=True)
            if dl_resp.status_code == 200 and dl_resp.content:
                meta = cache_manager.get_metadata(str(file_id))
                release_name = meta.get("release_name", file_id) if meta else file_id
                sub_bytes = transcode_to_utf8(dl_resp.content)
                await cache_manager.save_subtitle(str(file_id), sub_bytes)
                return _build_subtitle_response(
                    sub_bytes,
                    release_name,
                    req_fmt,
                    rtl_fix_enabled,
                    ad_removal_enabled,
                    keep_credits_enabled,
                    clean_options,
                    strip_hi_enabled,
                    eastern_numerals_enabled,
                    strip_diacritics_enabled,
                    convert_ass_enabled,
                )
        except Exception as dl_err:
            logger.warning(f"[OpenSubtitles Download] Error: {dl_err}")

    meta = cache_manager.get_metadata(str(file_id))
    if meta and meta.get("imdb_id"):
        effective_subsource_key = subsource_key or meta.get("subsource_key")
        fallback_bytes = await _fallback_download_subsource(
            imdb_id=meta["imdb_id"],
            media_type=meta.get("media_type") or ("series" if meta.get("season") is not None else "movie"),
            season=meta.get("season"),
            episode=meta.get("episode"),
            subsource_key=effective_subsource_key,
            target_filename=meta.get("target_filename") or meta.get("release_name"),
            lang=meta.get("lang", "ara"),
            client=client,
        )
        if fallback_bytes:
            await cache_manager.save_subtitle(str(file_id), fallback_bytes)
            return _build_subtitle_response(
                fallback_bytes,
                meta.get("release_name", file_id),
                req_fmt,
                rtl_fix_enabled,
                ad_removal_enabled,
                keep_credits_enabled,
                clean_options,
                strip_hi_enabled,
                eastern_numerals_enabled,
                strip_diacritics_enabled,
                convert_ass_enabled,
            )

    raise HTTPException(status_code=404, detail="Subtitle link expired or limit reached")


@app.api_route("/sub/{sub_id}.srt", methods=["GET", "HEAD"])
async def serve_subtitle(sub_id: str):
    return await _serve_subtitle_handler(sub_id, req_format="srt")


@app.api_route("/sub/{sub_id}.ass", methods=["GET", "HEAD"])
async def serve_subtitle_ass(sub_id: str):
    return await _serve_subtitle_handler(sub_id, req_format="ass")


@app.api_route("/sub/{sub_id}.ssa", methods=["GET", "HEAD"])
async def serve_subtitle_ssa(sub_id: str):
    return await _serve_subtitle_handler(sub_id, req_format="ssa")


@app.api_route("/sub/{sub_id}.vtt", methods=["GET", "HEAD"])
async def serve_subtitle_vtt(sub_id: str):
    return await _serve_subtitle_handler(sub_id, req_format="vtt")


@app.api_route("/{config}/sub/{sub_id}.srt", methods=["GET", "HEAD"])
async def serve_configured_subtitle(config: str, sub_id: str):
    return await _serve_subtitle_handler(sub_id, config_str=config, req_format="srt")


@app.api_route("/{config}/sub/{sub_id}.ass", methods=["GET", "HEAD"])
async def serve_configured_subtitle_ass(config: str, sub_id: str):
    return await _serve_subtitle_handler(sub_id, config_str=config, req_format="ass")


@app.api_route("/{config}/sub/{sub_id}.ssa", methods=["GET", "HEAD"])
async def serve_configured_subtitle_ssa(config: str, sub_id: str):
    return await _serve_subtitle_handler(sub_id, config_str=config, req_format="ssa")


@app.api_route("/{config}/sub/{sub_id}.vtt", methods=["GET", "HEAD"])
async def serve_configured_subtitle_vtt(config: str, sub_id: str):
    return await _serve_subtitle_handler(sub_id, config_str=config, req_format="vtt")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "env_keys": {
            "subdl": bool(settings.SUBDL_API_KEY.strip()),
            "subsource": bool(settings.SUBSOURCE_API_KEY.strip()),
            "opensubtitles": bool(settings.OPENSUBTITLES_API_KEY.strip()),
        },
        "cache": cache_manager.get_stats(),
    }


@app.api_route("/cache/clear", methods=["GET", "POST"])
async def clear_cache_route():
    clear_subtitle_cache()
    return {"status": "ok", "message": "In-memory TTLCache successfully cleared."}
