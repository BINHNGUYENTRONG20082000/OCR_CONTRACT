import model_cache  # noqa: F401
import os
from pathlib import Path

import onnxruntime as ort

# Load CUDA 12 + cuDNN DLLs from nvidia-* pip packages before creating the session.
ort.preload_dlls(directory="")

# cuDNN lazy-loads extra engine DLLs; put their folders on PATH first.
try:
    import nvidia

    nvidia_root = Path(next(iter(nvidia.__path__)))
    for bin_dir in nvidia_root.glob("*/bin"):
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_medium_det",
    text_recognition_model_name="PP-OCRv6_medium_rec",
    engine="onnxruntime",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
result = ocr.predict("E:/OCR_CONTRACT/Screenshot 2026-06-03 133248.jpg")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")