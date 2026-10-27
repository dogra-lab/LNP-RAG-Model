"""OCR fallback for scanned or image-only PDF pages (Tesseract via pytesseract)."""
from __future__ import annotations

import io
import logging
import shutil
from functools import lru_cache

import pymupdf

log = logging.getLogger(__name__)


@lru_cache
def tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # pragma: no cover - depends on system install
        return False


def ocr_page(page: pymupdf.Page, dpi: int = 300, lang: str = "eng") -> str:
    """Render a page to an image and OCR it. Returns "" if OCR is unavailable."""
    if not tesseract_available():
        return ""
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        # --psm 1: automatic page segmentation with orientation detection,
        # which copes with two-column layouts better than the default.
        return pytesseract.image_to_string(image, lang=lang, config="--psm 1")
    except Exception as exc:
        log.warning("ocr_failed", extra={"page": page.number + 1, "error": str(exc)})
        return ""
