"""Pipeline đầy đủ: từ file hợp đồng (PDF/ảnh/docx/md) -> OCR -> trích xuất thông tin -> JSON.

Gói gọn trong 1 class `ContractExtractor`:
    file hợp đồng  ──►  to_markdown()  ──►  extract_fields()  ──►  dict JSON

Các trường trích xuất:
    lương, ngày hiệu lực, chức vụ, loại hợp đồng, BHXH, MST cá nhân, nơi cấp/ngày cấp CCCD,
    địa chỉ thường trú, địa chỉ hiện tại, thời hạn hợp đồng, ngày kết thúc, vị trí công việc,
    số hợp đồng, phụ cấp (xăng xe, điện thoại, ăn trưa, trách nhiệm, HQCV),
    thưởng năng lực, tổng thu nhập, quản lý trực tiếp, phòng ban.

Tái sử dụng:
    - OCR             : scan_document.py + PP-OCRv6
    - Trích xuất field: extract_contract_fields.py (regex + LLM)

Chạy trực tiếp:
    python main.py                       # dùng DEFAULT_INPUT trong CONFIG
    python main.py input/hopdong.pdf     # truyền file qua CLI
    python main.py input/foo.md --no-llm # chỉ regex
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import extract_contract_fields as ecf
import ocr_engine
import scan_document as sd

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ====================================================================
# CONFIG - SỬA Ở ĐÂY ĐỂ CHẠY TRỰC TIẾP (python main.py)
# ====================================================================
DEFAULT_INPUT = "input/20260106212919160.pdf"   # file hợp đồng cần trích xuất

CUDA_DEVICE = "0"                                # GPU dùng cho PP-OCRv6 (máy 1 GPU = "0")

# Giới hạn trang OCR cho file lớn (thông tin hợp đồng chủ yếu ở đầu & cuối):
#   - PDF <= FULL_PAGE_THRESHOLD trang  -> OCR toàn bộ
#   - PDF >  FULL_PAGE_THRESHOLD trang  -> chỉ OCR HEAD_PAGES trang đầu + TAIL_PAGES trang cuối
FULL_PAGE_THRESHOLD = 5
HEAD_PAGES = 2
TAIL_PAGES = 2
OCR_DPI = 200              # render PDF (200 ~ đủ; 300 chậm hơn vì ảnh lớn ~14x so với screenshot)
OCR_MAX_LONG_EDGE = 1800   # thu nhỏ cạnh dài ảnh trước Paddle infer

# True = unload model sau mỗi file (tiết kiệm VRAM, lần sau phải nạp lại — chậm).
# False = giữ model trong VRAM, chỉ xóa cache infer (nhanh hơn cho nhiều file).
RELEASE_MODELS_AFTER_JOB = False

USE_LLM = True                                   # bật trích xuất bằng LLM
USE_OCR_CORRECT = True                           # LLM sửa dấu/chính tả sau OCR (ảnh/PDF scan)
LLM_MODEL = os.getenv("MODEL_NAME_CORE", "")     # để "" nếu server tự chọn model
LLM_API_URL = os.getenv("LLM_API_URL_CORE", "http://10.0.99.116:8070/v1/chat/completions")
LLM_STREAM = False
LLM_MAX_MARKDOWN_CHARS = 32000   # markdown gửi LLM; PDF ~20k thường không bị cắt

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
TEXT_EXT = {".md", ".txt"}


class ContractExtractor:
    """Pipeline trích xuất thông tin hợp đồng lao động từ file -> JSON."""

    def __init__(self,
                 device=CUDA_DEVICE,
                 use_llm=USE_LLM,
                 use_ocr_correct=USE_OCR_CORRECT,
                 llm_model=LLM_MODEL,
                 llm_api_url=LLM_API_URL,
                 llm_stream=LLM_STREAM,
                 full_page_threshold=FULL_PAGE_THRESHOLD,
                 head_pages=HEAD_PAGES,
                 tail_pages=TAIL_PAGES,
                 ocr_dpi=OCR_DPI,
                 ocr_max_long_edge=OCR_MAX_LONG_EDGE,
                 llm_max_markdown_chars=LLM_MAX_MARKDOWN_CHARS,
                 release_models_after_job=RELEASE_MODELS_AFTER_JOB):
        self.device = device
        self.use_llm = use_llm
        self.use_ocr_correct = use_ocr_correct
        self.llm_model = llm_model
        self.llm_stream = llm_stream
        self.full_page_threshold = full_page_threshold
        self.head_pages = head_pages
        self.tail_pages = tail_pages
        self.ocr_dpi = ocr_dpi
        self.ocr_max_long_edge = ocr_max_long_edge
        self.llm_max_markdown_chars = llm_max_markdown_chars
        self.release_models_after_job = release_models_after_job

        # Đồng bộ cấu hình LLM sang module extract_contract_fields
        ecf.LLM_API_URL = llm_api_url
        ecf.LLM_MAX_MARKDOWN_CHARS = llm_max_markdown_chars
        if llm_model:
            ecf.MODEL_NAME = llm_model

        # Các tài nguyên nặng -> load lazy khi cần
        self._engine = None
        self._converter = None

    # ---------- LAZY LOADERS ----------

    def _ensure_ocr(self):
        """Khởi tạo PP-OCRv6, chỉ 1 lần."""
        if self._engine is None:
            self._engine = ocr_engine.get_engine(
                device=self.device,
                max_long_edge=self.ocr_max_long_edge,
            )
        return self._engine

    def _ensure_converter(self):
        """Nạp DocumentConverter của docling (cho PDF text/docx/xlsx...)."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter

    # ---------- BƯỚC 1: FILE -> MARKDOWN ----------

    def _ocr_image(self, img_path):
        """OCR một file ảnh thành markdown."""
        engine = self._ensure_ocr()
        sd.auto_rotate(img_path)
        text = engine.ocr_image(str(img_path))
        text = sd.clean_empty_tables(text)
        text = sd.format_markdown_elements(text)
        return text

    def correct_ocr_markdown(self, markdown):
        """Sửa dấu/chính tả OCR bằng LLM (chỉ khi bật)."""
        if not self.use_ocr_correct:
            return markdown
        if not self.use_llm:
            logger.info("Bỏ qua sửa OCR (use_llm=False).")
            return markdown
        logger.info("Đang sửa chính tả OCR bằng LLM...")
        return ecf.llm_correct_ocr(
            markdown,
            model=self.llm_model,
            stream=self.llm_stream,
            max_chars=self.llm_max_markdown_chars,
        )

    def to_markdown(self, file_path):
        """Chuyển file hợp đồng (PDF/ảnh/docx/xlsx/md...) sang markdown."""
        path = Path(file_path)
        ext = path.suffix.lower()
        used_ocr = False

        if ext in TEXT_EXT:
            markdown = path.read_text(encoding="utf-8")
        elif ext in IMAGE_EXT:
            markdown = self._ocr_image(str(path))
            used_ocr = True
        else:
            if ext == ".doc":
                file_path = str(sd.convert_doc_to_docx(file_path))
                ext = ".docx"

            if ext == ".pdf":
                if sd.is_pdf_scan(file_path):
                    engine = self._ensure_ocr()
                    with tempfile.TemporaryDirectory() as tmp:
                        markdown = sd.convert_pdf_scan(
                            file_path, engine, tmp,
                            full_page_threshold=self.full_page_threshold,
                            head_pages=self.head_pages,
                            tail_pages=self.tail_pages,
                            dpi=self.ocr_dpi,
                            max_long_edge=self.ocr_max_long_edge,
                        )
                    used_ocr = True
                else:
                    try:
                        markdown = sd.convert_docling(file_path, self._ensure_converter())
                    except Exception as exc:
                        logger.warning("Docling lỗi (%s), fallback pdfplumber.", exc)
                        markdown = sd.convert_pdf_text(file_path)
            elif ext == ".docx":
                markdown = sd.convert_docx(file_path)
            elif ext in {".xlsx", ".xls"}:
                markdown = sd.convert_xlsx(file_path)
            else:
                try:
                    markdown = sd.convert_docling(file_path, self._ensure_converter())
                except Exception as exc:
                    logger.warning("Docling lỗi (%s).", exc)
                    raise

        if used_ocr:
            markdown = self.correct_ocr_markdown(markdown)
        return markdown

    # ---------- BƯỚC 2: MARKDOWN -> FIELDS ----------

    def extract_fields(self, markdown):
        """Trích xuất schema field từ markdown (regex + LLM rồi merge)."""
        rx_data = ecf.regex_extract(markdown)
        llm_data = None
        if self.use_llm:
            missing = ecf.count_missing(rx_data)
            if missing == 0:
                logger.info("Regex đã đủ field -> bỏ qua LLM (nhanh hơn).")
            else:
                logger.info("Regex còn %d field trống -> gọi LLM.", missing)
                slim = ecf.trim_markdown_for_llm(markdown, max_chars=self.llm_max_markdown_chars)
                llm_data = ecf.llm_extract(slim, self.llm_model, stream=self.llm_stream)
                if llm_data is None:
                    logger.warning("LLM không trả kết quả -> dùng regex-only.")
        return ecf.merge(llm_data, rx_data)

    def prewarm_ocr(self):
        """Nạp PP-OCRv6 sẵn khi mở app (tránh chờ lần upload đầu)."""
        self._ensure_ocr()
        logger.info("Đã pre-warm PP-OCRv6")

    # ---------- PIPELINE ĐẦY ĐỦ ----------

    def run(self, file_path, save=True):
        """Chạy toàn bộ pipeline cho 1 file, trả về dict; tùy chọn lưu .fields.json."""
        logger.info("Đang xử lý: %s", file_path)
        try:
            markdown = self.to_markdown(file_path)
            data = self.extract_fields(markdown)
            if save:
                out_path = Path(file_path).with_suffix(".fields.json")
                out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("Đã lưu: %s", out_path)
            return data
        finally:
            if self.release_models_after_job:
                self.close()
            else:
                self.release_cache()

    def release_cache(self):
        """Giữ model, chỉ xóa cache CUDA/activation sau infer."""
        ocr_engine.clear_cuda_cache()

    def close(self):
        """Giải phóng OCR + docling khỏi RAM/VRAM."""
        if self._engine is not None:
            if hasattr(self._engine, "close"):
                try:
                    self._engine.close()
                except Exception as exc:
                    logger.warning("Lỗi khi close OCR engine: %s", exc)
            self._engine = None

        if self._converter is not None:
            try:
                close_fn = getattr(self._converter, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
            self._converter = None

        ocr_engine.clear_cuda_cache()
        logger.info("Đã giải phóng tài nguyên OCR/docling")


def main():
    parser = argparse.ArgumentParser(description="Pipeline trích xuất thông tin hợp đồng -> JSON.")
    parser.add_argument("inputs", nargs="*", default=[DEFAULT_INPUT],
                        help="File hợp đồng (mặc định DEFAULT_INPUT trong code)")
    parser.add_argument("--no-llm", action="store_true", help="Chỉ dùng regex, không gọi LLM")
    parser.add_argument("--model", default=LLM_MODEL, help="Tên model LLM (để trống nếu server tự chọn)")
    args = parser.parse_args()

    inputs = args.inputs or [DEFAULT_INPUT]
    extractor = ContractExtractor(use_llm=USE_LLM and not args.no_llm, llm_model=args.model)

    results = {}
    try:
        for file_path in inputs:
            if not Path(file_path).is_file():
                logger.warning("Bỏ qua (không tìm thấy): %s", file_path)
                continue
            results[file_path] = extractor.run(file_path)
    finally:
        extractor.close()

    payload = next(iter(results.values())) if len(results) == 1 else results
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
