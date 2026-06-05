"""OCR bằng PaddleOCR-VL (paddleocr 3.x).

    engine = get_engine(device="1")
    engine.ocr_image(img_path) -> str   # markdown thô (chưa hậu xử lý)

Hậu xử lý (clean_empty_tables / format_markdown_elements) do scan_document đảm nhiệm.
"""

import logging
import os
import tempfile

logger = logging.getLogger(__name__)


class PaddleVLEngine:
    """OCR bằng PaddleOCR-VL. Lazy-load pipeline khi cần."""

    name = "paddle"

    def __init__(self, device="1", pipeline_version="v1", max_long_edge=1800, **kwargs):
        self.device = str(device)
        self.pipeline_version = pipeline_version
        self.max_long_edge = max_long_edge
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        logger.info("Đang nạp PaddleOCR-VL (pipeline_version=%s)...", self.pipeline_version)
        from paddleocr import PaddleOCRVL
        self._pipe = PaddleOCRVL(pipeline_version=self.pipeline_version)

    def _prepare_image_path(self, img_path):
        """Thu nhỏ ảnh quá lớn (PDF render) trước infer."""
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge <= self.max_long_edge:
            return str(img_path)
        scale = self.max_long_edge / long_edge
        nw, nh = int(w * scale), int(h * scale)
        logger.info("Thu nhỏ ảnh OCR %dx%d -> %dx%d", w, h, nw, nh)
        img = img.resize((nw, nh), Image.LANCZOS)
        out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(out.name, "JPEG", quality=92)
        return out.name

    def ocr_image(self, img_path):
        """OCR 1 ảnh -> markdown thô."""
        self._ensure()
        path = self._prepare_image_path(img_path)
        results = list(self._pipe.predict(path))
        parts = []
        for r in results:
            md = getattr(r, "markdown", None)
            if isinstance(md, dict):
                parts.append(md.get("markdown_texts", "") or "")
            else:
                parts.append(str(md or ""))
        return "\n\n".join(p for p in parts if p).strip()

    def close(self):
        self._pipe = None


def get_engine(device="1", pipeline_version="v1", max_long_edge=1800, **_ignored):
    """Trả về engine PaddleOCR-VL."""
    return PaddleVLEngine(
        device=device,
        pipeline_version=pipeline_version,
        max_long_edge=max_long_edge,
    )
