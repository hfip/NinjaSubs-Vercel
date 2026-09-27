"""Automated LRU Disk Cache Manager for subtitles (Serverless & Long Token Safe)."""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def _safe_filename(key: str) -> str:
    """Normalize and hash sub_id if too long to prevent Errno 36 File name too long."""
    clean_id = os.path.basename(key).replace("..", "").strip()
    if len(clean_id) > 64:
        return hashlib.md5(clean_id.encode("utf-8")).hexdigest()
    return clean_id


class LRUCacheManager:
    """
    Manages cached .srt files on disk with automated LRU cleanup.
    Fully compatible with Linux file systems and Vercel /tmp sandbox.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        max_bytes: int = settings.CACHE_MAX_BYTES,
        max_files: int = settings.CACHE_MAX_FILES,
    ):
        target_dir = cache_dir or settings.CACHE_DIR
        # Fallback to /tmp if running on serverless or path is read-only
        if os.getenv("VERCEL") or not os.access(Path(target_dir).parent, os.W_OK):
            target_dir = "/tmp/subs_cache"

        self.cache_dir = Path(target_dir)
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.meta_dir = self.cache_dir / "_meta"
        self._lock = asyncio.Lock()

        # Ensure directories exist safely
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.meta_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Failed to create cache directory {self.cache_dir}: {e}")

    def get_subtitle_path(self, sub_id: str) -> Path:
        """Return safe path for a given subtitle ID."""
        safe_name = _safe_filename(sub_id)
        return self.cache_dir / f"{safe_name}.srt"

    def get_meta_path(self, sub_id: str) -> Path:
        """Return safe path for subtitle metadata JSON."""
        safe_name = _safe_filename(sub_id)
        return self.meta_dir / f"{safe_name}.json"

    async def get_subtitle(self, sub_id: str) -> bytes | None:
        """Retrieve cached subtitle file if it exists."""
        try:
            file_path = self.get_subtitle_path(sub_id)
            if not file_path.is_file():
                return None

            async with self._lock:
                try:
                    now = time.time()
                    try:
                        os.utime(file_path, (now, now))
                    except OSError:
                        pass
                    return file_path.read_bytes()
                except OSError as e:
                    logger.warning(f"Error reading cached subtitle {file_path}: {e}")
                    return None
        except OSError:
            return None

    async def save_subtitle(self, sub_id: str, data: bytes) -> bool:
        """Atomically save subtitle file to disk and trigger LRU cleanup."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.get_subtitle_path(sub_id)
            tmp_path = file_path.with_suffix(".srt.tmp")

            async with self._lock:
                try:
                    tmp_path.write_bytes(data)
                    tmp_path.replace(file_path)
                    now = time.time()
                    try:
                        os.utime(file_path, (now, now))
                    except OSError:
                        pass
                except OSError as e:
                    logger.error(f"Failed to write subtitle {file_path}: {e}")
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
                    return False

                self._enforce_limits()
                return True
        except Exception as e:
            logger.error(f"Save subtitle failed: {e}")
            return False

    def store_metadata(self, sub_id: str, metadata: dict[str, Any]) -> None:
        """Store download metadata safely."""
        try:
            self.meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = self.get_meta_path(sub_id)
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        except OSError as e:
            logger.warning(f"Failed to store metadata for {sub_id}: {e}")

    def get_metadata(self, sub_id: str) -> dict[str, Any] | None:
        """Retrieve stored download metadata."""
        try:
            meta_path = self.get_meta_path(sub_id)
            if not meta_path.is_file():
                return None
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read metadata for {sub_id}: {e}")
            return None

    def clear_metadata(self) -> None:
        """Invalidate and wipe all on-disk metadata cache entries."""
        try:
            if self.meta_dir.is_dir():
                for p in self.meta_dir.glob("*.json"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"Failed to clear metadata directory {self.meta_dir}: {e}")

    def _enforce_limits(self) -> None:
        """Enforce max_bytes and max_files via LRU eviction."""
        try:
            entries: list[Path] = [
                p for p in self.cache_dir.iterdir() if p.is_file() and p.suffix == ".srt"
            ]
        except OSError:
            return

        if not entries:
            return

        file_stats = []
        total_size = 0
        for p in entries:
            try:
                stat = p.stat()
                atime = getattr(stat, "st_atime", stat.st_mtime)
                file_stats.append((atime, stat.st_size, p))
                total_size += stat.st_size
            except OSError:
                continue

        if len(file_stats) <= self.max_files and total_size <= self.max_bytes:
            return

        file_stats.sort(key=lambda item: item[0])
        current_count = len(file_stats)
        for _atime, size, p in file_stats:
            if current_count <= self.max_files and total_size <= self.max_bytes:
                break
            try:
                p.unlink()
                meta_p = self.get_meta_path(p.stem)
                if meta_p.exists():
                    meta_p.unlink()
                total_size -= size
                current_count -= 1
            except OSError:
                pass

    def get_stats(self) -> dict[str, Any]:
        """Return cache health and usage statistics."""
        try:
            entries = [p for p in self.cache_dir.iterdir() if p.is_file() and p.suffix == ".srt"]
            total_size = sum(p.stat().st_size for p in entries)
            return {
                "cache_dir": str(self.cache_dir),
                "file_count": len(entries),
                "max_files": self.max_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "max_bytes": self.max_bytes,
                "max_mb": round(self.max_bytes / (1024 * 1024), 2),
            }
        except Exception as e:
            return {"error": str(e)}


cache_manager = LRUCacheManager()
