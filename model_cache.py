"""Cấu hình thư mục model local cho toàn project.

Lần đầu: tải model về ./model/
Lần sau: đọc từ ./model/ (không tải lại nếu đã có).

Phải import module này TRƯỚC khi import paddleocr / docling / transformers.
Có thể ghi đè bằng biến môi trường OCR_CONTRACT_MODEL_DIR.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("OCR_CONTRACT_MODEL_DIR", _ROOT / "model")).resolve()

_PADDLEX_DIR = MODEL_DIR / "paddlex"
_HF_DIR = MODEL_DIR / "huggingface"
_HF_HUB_DIR = _HF_DIR / "hub"
_HF_TRANSFORMERS_DIR = _HF_DIR / "transformers"

_CONFIGURED = False


def setup_model_cache(*, force: bool = False) -> Path:
    """Tạo thư mục model và gán env cache. Gọi an toàn nhiều lần."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return MODEL_DIR

    for path in (_PADDLEX_DIR, _HF_HUB_DIR, _HF_TRANSFORMERS_DIR):
        path.mkdir(parents=True, exist_ok=True)

    # setdefault: tôn trọng env người dùng đã set sẵn
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_PADDLEX_DIR))
    os.environ.setdefault("HF_HOME", str(_HF_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_HUB_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_TRANSFORMERS_DIR))
    # Bỏ check host model mỗi lần khởi động (vẫn tải nếu thiếu file local)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    _CONFIGURED = True
    logger.info(
        "Model cache: %s (paddlex=%s, hf=%s)",
        MODEL_DIR,
        os.environ["PADDLE_PDX_CACHE_HOME"],
        os.environ["HUGGINGFACE_HUB_CACHE"],
    )
    return MODEL_DIR


# Tự cấu hình khi import
setup_model_cache()
