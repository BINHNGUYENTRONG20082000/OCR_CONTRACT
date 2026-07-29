# OCR_CONTRACT

Pipeline trích xuất thông tin **hợp đồng lao động tiếng Việt** từ PDF / ảnh / DOCX / Markdown → JSON có cấu trúc.

```
File hợp đồng
    → Đọc / OCR (PP-OCRv6 hoặc docling)
    → (tuỳ chọn) LLM sửa chính tả OCR
    → Regex + LLM trích field
    → JSON / giao diện Gradio
```

---

## Tính năng chính

- Hỗ trợ nhiều định dạng: PDF (scan & text), ảnh, DOCX, DOC, XLSX, Markdown
- **PDF scan / ảnh**: OCR bằng **PP-OCRv6** (ONNX Runtime, GPU nếu có)
- **PDF text**: đọc bằng **docling** (fallback `pdfplumber`)
- PDF lớn (> 5 trang): chỉ OCR 2 trang đầu + 2 trang cuối (có thể chỉnh)
- Trích xuất lai **regex + LLM** (regex ưu tiên số HĐ, CCCD, BHXH, ngày…)
- Tuỳ chọn **LLM sửa dấu/chính tả** sau OCR (PP-OCR thường rớt dấu tiếng Việt)
- UI Gradio: bảng field, JSON, markdown trung gian, tải file kết quả
- Model AI cache vào thư mục `./model/` (lần sau không tải lại)

### Các trường trích xuất

| Nhóm | Field |
|------|--------|
| Hợp đồng | `so_hop_dong`, `loai_hop_dong`, `thoi_han_hop_dong`, `ngay_hieu_luc`, `ngay_ket_thuc` |
| Công việc | `chuc_vu`, `vi_tri_cong_viec`, `phong_ban`, `quan_ly_truc_tiep` |
| Cá nhân | `dia_chi_thuong_tru`, `dia_chi_hien_tai`, `cccd` (so/ngay_cap/noi_cap), `ma_so_bhxh`, `mst_ca_nhan` |
| Lương | `luong_co_ban`, `phu_cap` (xang_xe, dien_thoai, an_trua, trach_nhiem, hqcv), `thuong_nang_luc`, `tong_thu_nhap` |

---

## Cấu trúc project

```
OCR_CONTRACT/
├── contract_app.py              # Giao diện Gradio
├── main.py                      # Pipeline CLI + ContractExtractor
├── ocr_engine.py                # PP-OCRv6 (ONNX)
├── scan_document.py             # Convert file → markdown
├── extract_contract_fields.py   # Regex + LLM trích field / sửa OCR
├── model_cache.py               # Trỏ cache model về ./model/
├── requirements.txt
├── model/                       # Cache model (tự tạo khi chạy)
│   ├── paddlex/                 # PP-OCRv6
│   └── huggingface/             # Docling
├── input/                       # Đặt file mẫu (tuỳ chọn)
└── output/                      # Kết quả thử nghiệm OCR (tuỳ chọn)
```

---

## Yêu cầu hệ thống

| Thành phần | Ghi chú |
|------------|---------|
| Python | 3.10 – 3.12 khuyến nghị |
| GPU NVIDIA (tuỳ chọn) | Driver CUDA 12.x; RTX khuyến nghị. Không có GPU: dùng `onnxruntime` (CPU) |
| Poppler | Bắt buộc cho `pdf2image` (render PDF scan) |
| Tesseract OCR | Dùng OSD xoay ảnh (`pytesseract`) |
| LibreOffice | Chỉ cần nếu xử lý file `.doc` cũ (`soffice`) |
| LLM API | Endpoint tương thích chat completions (OpenAI/Ollama-style) |

---

## Cài đặt

### 1. Clone / mở project

```bash
cd OCR_CONTRACT
```

### 2. Tạo môi trường ảo (khuyến nghị Conda)

```bash
conda create -n paddleocr python=3.12 -y
conda activate paddleocr
```

Hoặc `venv`:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. Cài Python packages

**Có GPU NVIDIA (mặc định trong `requirements.txt`):**

```bash
pip install -r requirements.txt
```

**Chỉ CPU:** sửa `requirements.txt` — thay `onnxruntime-gpu` bằng `onnxruntime`, và có thể bỏ các dòng `nvidia-*-cu12`, rồi:

```bash
pip install -r requirements.txt
```

### 4. Cài công cụ hệ thống

#### Windows

1. **Poppler**  
   - Tải bản Windows (vd. từ [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases))  
   - Giải nén và thêm thư mục `bin` vào `PATH`

2. **Tesseract**  
   - Cài từ [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)  
   - Thêm vào `PATH` (vd. `C:\Program Files\Tesseract-OCR`)  
   - Nên cài thêm ngôn ngữ `vie` nếu dùng OSD/OCR phụ

3. **LibreOffice** (tuỳ chọn, cho `.doc`)  
   - Cài và đảm bảo lệnh `soffice` gọi được từ terminal

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y poppler-utils tesseract-ocr tesseract-ocr-vie libreoffice
```

### 5. Cấu hình LLM

Mặc định trong `main.py`:

```text
http://10.0.99.116:8070/v1/chat/completions
```

Đổi bằng biến môi trường (khuyến nghị) hoặc sửa trực tiếp `main.py`:

**Windows PowerShell:**

```powershell
$env:LLM_API_URL_CORE = "http://<host>:<port>/v1/chat/completions"
$env:MODEL_NAME_CORE = "<ten_model>"   # để trống nếu server tự chọn
```

**Linux / macOS:**

```bash
export LLM_API_URL_CORE="http://<host>:<port>/v1/chat/completions"
export MODEL_NAME_CORE="<ten_model>"
```

API cần hỗ trợ `POST` chat completions (format OpenAI hoặc tương đương).

### 6. GPU device

Trong `main.py`:

```python
CUDA_DEVICE = "0"   # máy 1 GPU = "0"
```

Nếu set sai (vd. `"1"` khi chỉ có GPU 0), ONNXRuntime sẽ báo không thấy CUDA và fallback CPU.

---

## Chạy ứng dụng

### Giao diện Gradio (khuyến nghị)

```bash
python contract_app.py
```

- Local: [http://127.0.0.1:7875](http://127.0.0.1:7875)  
- Có thể có link `share` Gradio (public tạm thời)

Trên UI:

- Upload PDF / ảnh / DOCX / MD  
- **Dùng LLM trích field** — tắt = chỉ regex  
- **LLM sửa chính tả OCR** — chỉ áp dụng ảnh / PDF scan  

### CLI pipeline đầy đủ

```bash
# File mặc định trong main.py (DEFAULT_INPUT)
python main.py

# Truyền file
python main.py path/toi/hopdong.pdf

# Chỉ regex, không gọi LLM
python main.py path/toi/hopdong.pdf --no-llm

# Chỉ định model LLM
python main.py path/toi/hopdong.pdf --model <ten_model>
```

Kết quả: in JSON ra stdout và (mặc định) ghi `<stem>.fields.json` cạnh file đầu vào.

### Chỉ trích field từ markdown có sẵn

```bash
python extract_contract_fields.py path/toi/file.md
python extract_contract_fields.py path/toi/file.md --no-llm
```

---

## Cache model (`./model/`)

Lần chạy đầu, model được tải về:

| Thư mục | Nội dung |
|---------|----------|
| `model/paddlex/` | PP-OCRv6 det/rec (ONNX) |
| `model/huggingface/` | Docling layout / tableformer |

Lần sau đọc từ đây, không tải lại nếu đã có file.

Đổi vị trí cache:

```powershell
$env:OCR_CONTRACT_MODEL_DIR = "D:\my_models"
```

```bash
export OCR_CONTRACT_MODEL_DIR=/data/ocr_models
```

Các file model lớn **không** commit git (đã ignore).

---

## Cấu hình quan trọng (`main.py`)

| Biến | Ý nghĩa | Mặc định |
|------|---------|----------|
| `CUDA_DEVICE` | GPU index cho OCR | `"0"` |
| `FULL_PAGE_THRESHOLD` | PDF ≤ N trang → OCR hết | `5` |
| `HEAD_PAGES` / `TAIL_PAGES` | PDF lớn: trang đầu/cuối OCR | `2` / `2` |
| `OCR_DPI` | DPI render PDF scan | `200` |
| `OCR_MAX_LONG_EDGE` | Thu nhỏ cạnh dài ảnh trước OCR | `1800` |
| `USE_LLM` | Bật LLM trích field | `True` |
| `USE_OCR_CORRECT` | LLM sửa chính tả sau OCR | `True` |
| `RELEASE_MODELS_AFTER_JOB` | `True` = unload model sau mỗi file (tiết kiệm VRAM, chậm hơn) | `False` |
| `LLM_API_URL` | Endpoint LLM | env / mặc định nội bộ |
| `LLM_MAX_MARKDOWN_CHARS` | Giới hạn markdown gửi LLM | `32000` |

---

## Luồng xử lý chi tiết

1. **Ảnh / PDF scan** → PP-OCRv6 → (tuỳ chọn) LLM sửa dấu → markdown  
2. **PDF text** → docling (hoặc pdfplumber) → markdown  
3. **DOCX / XLSX** → python-docx / openpyxl → markdown  
4. **Markdown** → regex extract → nếu thiếu field và bật LLM → LLM extract → merge → JSON  

---

## Xử lý lỗi thường gặp

| Triệu chứng | Cách xử lý |
|-------------|------------|
| `no CUDA-capable device` / fallback CPU | Đặt `CUDA_DEVICE = "0"`; kiểm `nvidia-smi`; cài đúng `onnxruntime-gpu` + driver |
| `pdf2image` / Poppler lỗi | Thêm Poppler `bin` vào `PATH`, mở lại terminal |
| `tesseract is not installed` | Cài Tesseract và thêm vào `PATH` |
| Docling chậm / tải HF mỗi lần | Bình thường lần đầu; lần sau dùng `model/huggingface/`. Offline: `HF_HUB_OFFLINE=1` sau khi đã cache |
| OCR thiếu dấu tiếng Việt | Bật **LLM sửa chính tả OCR**; hoặc dùng PDF text thay vì scan nếu có |
| VRAM tăng dần | Giữ `RELEASE_MODELS_AFTER_JOB = False` + cache clear; nếu cần giải phóng hẳn: `True` (mỗi lần sẽ nạp lại model) |
| LLM không trả JSON | Kiểm tra `LLM_API_URL_CORE`, model name, và log warning trong console |

---

## Phụ thuộc chính

Xem chi tiết trong [`requirements.txt`](requirements.txt):

- `paddleocr` + `onnxruntime-gpu` — OCR  
- `docling`, `pdfplumber`, `pdf2image`, `python-docx`, `openpyxl` — đọc tài liệu  
- `gradio`, `requests` — UI và LLM client  
- `pillow`, `numpy`, `opencv-contrib-python` — xử lý ảnh  

---

## Giấy phép / ghi chú

Project nội bộ phục vụ trích xuất hợp đồng lao động. Model PP-OCRv6 / Docling tuân theo giấy phép của nhà phát hành tương ứng (PaddlePaddle, Docling, Hugging Face).
