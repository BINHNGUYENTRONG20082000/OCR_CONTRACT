# Model cache (local)

Lần chạy đầu tiên, các model AI được tải vào đây:

| Thư mục | Nội dung |
|---------|----------|
| `paddlex/` | PP-OCRv6 (det/rec ONNX) qua PaddleX |
| `huggingface/` | Docling layout / tableformer (Hugging Face) |

Các lần sau đọc từ đây, không tải lại nếu file đã có.

Đổi vị trí: set env `OCR_CONTRACT_MODEL_DIR` trước khi chạy app.
