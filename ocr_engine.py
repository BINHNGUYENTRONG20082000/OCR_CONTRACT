"""OCR bằng PP-OCRv6 (paddleocr 3.x).

    engine = get_engine(device="0")
    engine.ocr_image(img_path) -> str   # text theo thứ tự đọc (chưa hậu xử lý)

Hậu xử lý (clean_empty_tables / format_markdown_elements) do scan_document đảm nhiệm.
"""

import gc
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _preload_onnx_cuda():
    """Nạp CUDA/cuDNN DLL từ gói nvidia-* trước khi tạo session ONNXRuntime."""
    try:
        import onnxruntime as ort

        ort.preload_dlls(directory="")
    except Exception:
        pass
    try:
        import nvidia

        nvidia_root = Path(next(iter(nvidia.__path__)))
        for bin_dir in nvidia_root.glob("*/bin"):
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def clear_cuda_cache():
    """Cố gắng trả VRAM cache về OS (PyTorch / CuPy nếu có)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _result_as_dict(res):
    """Chuẩn hóa OCR result object về dict (chỉ giữ field cần thiết)."""
    if isinstance(res, dict):
        src = res
    else:
        src = None
        for attr in ("json", "res"):
            if hasattr(res, attr):
                val = getattr(res, attr)
                val = val() if callable(val) else val
                if isinstance(val, dict):
                    inner = val.get("res") if "rec_texts" not in val and isinstance(val.get("res"), dict) else val
                    src = inner if isinstance(inner, dict) else val
                    break
        if src is None:
            try:
                src = {
                    "rec_texts": list(res["rec_texts"]),
                    "dt_polys": list(res["dt_polys"]),
                    "rec_scores": list(res.get("rec_scores", [])),
                }
            except Exception:
                src = {
                    "rec_texts": list(getattr(res, "rec_texts", None) or []),
                    "dt_polys": list(getattr(res, "dt_polys", None) or []),
                    "rec_scores": list(getattr(res, "rec_scores", None) or []),
                }

    # Copy nông sang list/python thuần — tránh giữ reference ảnh/tensor trong result object
    return {
        "rec_texts": list(src.get("rec_texts") or []),
        "dt_polys": [list(map(list, p)) for p in (src.get("dt_polys") or [])],
        "rec_scores": list(src.get("rec_scores") or []),
    }


def _poly_box(poly):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def boxes_to_text(texts, polys, scores=None, score_thresh=0.0):
    """Ghép box OCR thành text theo thứ tự đọc (trên→dưới, trái→phải)."""
    items = []
    n = min(len(texts), len(polys))
    for i in range(n):
        text = (texts[i] or "").strip()
        if not text:
            continue
        if scores is not None and i < len(scores):
            try:
                if float(scores[i]) < score_thresh:
                    continue
            except (TypeError, ValueError):
                pass
        x0, y0, x1, y1 = _poly_box(polys[i])
        items.append({
            "text": text,
            "x": x0,
            "y": (y0 + y1) / 2.0,
            "h": max(1.0, y1 - y0),
        })

    if not items:
        return ""

    items.sort(key=lambda it: (it["y"], it["x"]))
    lines = []
    for item in items:
        if not lines:
            lines.append([item])
            continue
        cur = lines[-1]
        avg_y = sum(x["y"] for x in cur) / len(cur)
        avg_h = sum(x["h"] for x in cur) / len(cur)
        if abs(item["y"] - avg_y) <= max(avg_h * 0.55, 12.0):
            cur.append(item)
        else:
            lines.append([item])

    out_lines = []
    for line in lines:
        line.sort(key=lambda it: it["x"])
        out_lines.append(" ".join(it["text"] for it in line))
    return "\n".join(out_lines)


def _dispose_paddle_obj(obj):
    """Gọi các hook close/release phổ biến trên object paddlex/paddleocr."""
    if obj is None:
        return
    for name in ("close", "release", "clear", "reset"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass
    # Một số pipeline giữ list session ONNX
    for attr in ("_models", "models", "_components", "components", "paddlex_pipeline"):
        inner = getattr(obj, attr, None)
        if inner is None:
            continue
        try:
            values = inner.values() if isinstance(inner, dict) else inner
            for child in values:
                _dispose_paddle_obj(child)
        except Exception:
            pass


class PPOCRV6Engine:
    """OCR bằng PP-OCRv6. Lazy-load pipeline khi cần."""

    name = "ppocrv6"

    def __init__(
        self,
        device="0",
        max_long_edge=1800,
        det_model="PP-OCRv6_medium_det",
        rec_model="PP-OCRv6_medium_rec",
        engine="onnxruntime",
        score_thresh=0.0,
        **kwargs,
    ):
        self.device = str(device)
        self.max_long_edge = max_long_edge
        self.det_model = det_model
        self.rec_model = rec_model
        self.engine = engine
        self.score_thresh = score_thresh
        self._pipe = None

    def _ensure(self):
        if self._pipe is not None:
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        if self.engine == "onnxruntime":
            _preload_onnx_cuda()
        logger.info(
            "Đang nạp PP-OCRv6 (det=%s, rec=%s, engine=%s)...",
            self.det_model, self.rec_model, self.engine,
        )
        from paddleocr import PaddleOCR

        kwargs = dict(
            text_detection_model_name=self.det_model,
            text_recognition_model_name=self.rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        if self.engine:
            kwargs["engine"] = self.engine
        self._pipe = PaddleOCR(**kwargs)

    def _prepare_image_path(self, img_path):
        """Thu nhỏ ảnh quá lớn (PDF render) trước infer. Trả (path, is_temp)."""
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge <= self.max_long_edge:
            return str(img_path), False
        scale = self.max_long_edge / long_edge
        nw, nh = int(w * scale), int(h * scale)
        logger.info("Thu nhỏ ảnh OCR %dx%d -> %dx%d", w, h, nw, nh)
        img = img.resize((nw, nh), Image.LANCZOS)
        out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(out.name, "JPEG", quality=92)
        img.close()
        return out.name, True

    def ocr_image(self, img_path):
        """OCR 1 ảnh -> text theo thứ tự đọc."""
        self._ensure()
        path, is_temp = self._prepare_image_path(img_path)
        results = None
        try:
            results = self._pipe.predict(path)
            parts = []
            for r in results:
                data = _result_as_dict(r)
                text = boxes_to_text(
                    data.get("rec_texts") or [],
                    data.get("dt_polys") or [],
                    scores=data.get("rec_scores"),
                    score_thresh=self.score_thresh,
                )
                if text:
                    parts.append(text)
            return "\n\n".join(parts).strip()
        finally:
            # Bỏ reference result (có thể giữ ảnh/tensor lớn) + file tạm resize
            if results is not None:
                try:
                    for r in results:
                        if hasattr(r, "__dict__"):
                            r.__dict__.clear()
                except Exception:
                    pass
                del results
            if is_temp:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            clear_cuda_cache()

    def close(self):
        """Hủy pipeline + session ONNX và cố gắng trả VRAM."""
        if self._pipe is not None:
            try:
                _dispose_paddle_obj(self._pipe)
            except Exception as exc:
                logger.debug("dispose paddle obj: %s", exc)
            try:
                del self._pipe
            except Exception:
                pass
            self._pipe = None
        clear_cuda_cache()
        logger.info("Đã giải phóng PP-OCRv6 khỏi bộ nhớ/VRAM")


def get_engine(
    device="0",
    max_long_edge=1800,
    det_model="PP-OCRv6_medium_det",
    rec_model="PP-OCRv6_medium_rec",
    engine="onnxruntime",
    score_thresh=0.0,
    **_ignored,
):
    """Trả về engine PP-OCRv6."""
    return PPOCRV6Engine(
        device=device,
        max_long_edge=max_long_edge,
        det_model=det_model,
        rec_model=rec_model,
        engine=engine,
        score_thresh=score_thresh,
    )
