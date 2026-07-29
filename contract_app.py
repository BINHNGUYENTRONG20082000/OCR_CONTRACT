"""Giao diện Gradio cho pipeline trích xuất thông tin hợp đồng lao động.

Người dùng upload file hợp đồng (PDF/ảnh/docx/md) -> bấm Trích xuất ->
xem kết quả dưới dạng bảng, JSON, và markdown OCR trung gian; có thể tải JSON.

Chạy:
    python contract_app.py
Mặc định mở tại http://0.0.0.0:7862
"""

import model_cache  # noqa: F401  — cấu hình ./model trước khi nạp paddle/docling

import json
import logging
import tempfile
import threading
from pathlib import Path

import gradio as gr

from main import ContractExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Khởi tạo pipeline 1 lần; pre-warm OCR nền để lần upload đầu không chờ nạp model
extractor = ContractExtractor()


def _prewarm_ocr_background():
    try:
        extractor.prewarm_ocr()
    except Exception as exc:
        logger.warning("Pre-warm OCR thất bại (sẽ nạp khi upload): %s", exc)


threading.Thread(target=_prewarm_ocr_background, daemon=True).start()

# Nhãn tiếng Việt + đường dẫn (dotted) tới giá trị trong schema, theo thứ tự hiển thị
FIELD_LABELS = [
    ("Số hợp đồng", "so_hop_dong"),
    ("Loại hợp đồng", "loai_hop_dong"),
    ("Thời hạn hợp đồng", "thoi_han_hop_dong"),
    ("Ngày hiệu lực", "ngay_hieu_luc"),
    ("Ngày kết thúc", "ngay_ket_thuc"),
    ("Chức vụ", "chuc_vu"),
    ("Vị trí công việc", "vi_tri_cong_viec"),
    ("Phòng ban", "phong_ban"),
    ("Quản lý trực tiếp", "quan_ly_truc_tiep"),
    ("Địa chỉ thường trú", "dia_chi_thuong_tru"),
    ("Địa chỉ hiện tại", "dia_chi_hien_tai"),
    ("Số CCCD", "cccd.so"),
    ("Ngày cấp CCCD", "cccd.ngay_cap"),
    ("Nơi cấp CCCD", "cccd.noi_cap"),
    ("Mã số BHXH", "ma_so_bhxh"),
    ("MST cá nhân", "mst_ca_nhan"),
    ("Lương cơ bản", "luong_co_ban"),
    ("Phụ cấp xăng xe", "phu_cap.xang_xe"),
    ("Phụ cấp điện thoại", "phu_cap.dien_thoai"),
    ("Phụ cấp ăn trưa", "phu_cap.an_trua"),
    ("Phụ cấp trách nhiệm", "phu_cap.trach_nhiem"),
    ("Phụ cấp HQCV", "phu_cap.hqcv"),
    ("Thưởng năng lực", "thuong_nang_luc"),
    ("Tổng thu nhập", "tong_thu_nhap"),
]


def _get(data, dotted):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def to_table(data):
    """Chuyển dict kết quả thành bảng [Trường, Giá trị] để hiển thị."""
    rows = []
    for label, path in FIELD_LABELS:
        val = _get(data, path)
        rows.append([label, "" if val in (None, "") else str(val)])
    return rows


def process(file_path, use_llm, use_ocr_correct, progress=gr.Progress()):
    """Hàm xử lý chính khi bấm nút Trích xuất."""
    if not file_path:
        return [], {}, "", None, "⚠️ Vui lòng upload file hợp đồng."

    extractor.use_llm = use_llm
    extractor.use_ocr_correct = use_ocr_correct
    markdown = ""
    data = None
    error = None
    try:
        progress(0.1, desc="Đang OCR / đọc nội dung...")
        markdown = extractor.to_markdown(file_path)

        progress(0.7, desc="Đang trích xuất thông tin...")
        data = extractor.extract_fields(markdown)
    except Exception as exc:
        error = exc
    finally:
        # Mặc định giữ model; chỉ xóa cache. Unload khi RELEASE_MODELS_AFTER_JOB=True.
        if extractor.release_models_after_job:
            extractor.close()
        else:
            extractor.release_cache()

    if error is not None:
        return [], {}, "", None, f"❌ Lỗi xử lý: {error}"

    # Ghi JSON ra file tạm để tải về
    out_dir = tempfile.mkdtemp()
    json_path = Path(out_dir) / f"{Path(file_path).stem}.fields.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return to_table(data), data, markdown, str(json_path), "✅ Hoàn tất."


with gr.Blocks(title="AIPT - Trích xuất hợp đồng", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📄 Trích xuất thông tin Hợp đồng lao động
        Upload file hợp đồng (PDF / ảnh / DOCX / Markdown) rồi bấm **Trích xuất**.
        File PDF > 5 trang: OCR 2 trang đầu + 2 trang cuối. Thời gian phụ thuộc **độ dày nội dung từng trang** (trang nhiều chữ ~30–40s, trang phụ lục ~3–8s), không chỉ số trang.
        """
    )

    with gr.Row():
        # ===== CỘT TRÁI: INPUT =====
        with gr.Column(scale=1):
            file_in = gr.File(
                label="File hợp đồng",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".docx", ".doc", ".md", ".txt"],
                type="filepath",
            )
            use_llm = gr.Checkbox(value=True, label="Dùng LLM trích field (tắt = chỉ regex)")
            use_ocr_correct = gr.Checkbox(
                value=True,
                label="LLM sửa chính tả OCR (ảnh/PDF scan — cần bật LLM)",
            )
            btn = gr.Button("🚀 Trích xuất", variant="primary", size="lg")
            status = gr.Markdown("")
            json_file = gr.File(label="Tải JSON", interactive=False)

        # ===== CỘT PHẢI: OUTPUT =====
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Bảng thông tin"):
                    table_out = gr.Dataframe(
                        headers=["Trường", "Giá trị"],
                        datatype=["str", "str"],
                        wrap=True,
                        interactive=False,
                    )
                with gr.Tab("JSON"):
                    json_out = gr.JSON(label="Kết quả")
                with gr.Tab("Markdown OCR"):
                    md_out = gr.Markdown("")

    btn.click(
        process,
        inputs=[file_in, use_llm, use_ocr_correct],
        outputs=[table_out, json_out, md_out, json_file, status],
    )


if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7875,
        share=True,
    )
