"""Shared download utilities for occulus example scripts.

Uses certifi for SSL certificate verification and a browser-like
user-agent to satisfy server requirements.
"""

from __future__ import annotations

import logging
import ssl
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 occulus-examples/1.0 (+https://github.com/chrislyonsKY/occulus)"


def _ssl_ctx() -> ssl.SSLContext:
    """Return an SSL context using certifi root certificates."""
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    return ctx


def download(url: str, dest: Path, label: str = "") -> Path:
    """Download *url* to *dest*, using certifi SSL and caching.

    Parameters
    ----------
    url : str
        Remote URL to download.
    dest : Path
        Destination file path (parent directory must exist).
    label : str
        Human-readable label for log messages.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    if dest.exists():
        logger.info("Using cached file: %s", dest.name)
        return dest
    msg = label or dest.name
    logger.info("Downloading %s…", msg)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        logger.info("  → %s (%.1f MB)", dest.name, len(data) / 1e6)
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local file.", exc)
        sys.exit(1)
    return dest


def fetch_json(url: str) -> dict:
    """Fetch JSON from *url* using certifi SSL and browser user-agent."""
    import json

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        logger.error("HTTP request failed: %s", exc)
        raise


def find_usgs_tile(bbox: str) -> str:
    """Query the USGS TNM API and return the download URL for the first tile.

    Parameters
    ----------
    bbox : str
        Bounding box as ``"west,south,east,north"`` in WGS84 decimal degrees.

    Returns
    -------
    str
        Download URL for the first matching LiDAR tile.
    """
    import urllib.parse

    params = urllib.parse.urlencode(
        {
            "datasets": "Lidar Point Cloud (LPC)",
            "bbox": bbox,
            "max": 1,
            "prodFormats": "LAZ",
        }
    )
    url = f"https://tnmaccess.nationalmap.gov/api/v1/products?{params}"
    data = fetch_json(url)
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"No LiDAR tiles found for bbox: {bbox}")
    return items[0]["downloadURL"]
