"""Trích xuất thông tin hợp đồng lao động từ file Markdown (kết quả OCR) ra JSON.

Chiến lược lai (hybrid):
  - REGEX  : xử lý các trường có format cố định (số HĐ, CCCD, ngày cấp, BHXH,
             MST cá nhân, ngày tháng, thời hạn) -> độ chính xác cao, ghi đè LLM.
  - LLM    : xử lý các trường ngữ nghĩa (chức vụ, phòng ban, quản lý trực tiếp,
             loại hợp đồng) và tiền lương/phụ cấp (xử lý được trường hợp gộp/thiếu).
  - MERGE  : gộp 2 kết quả, regex thắng ở nhóm ID/ngày, LLM bù cho phần còn lại.

Cách dùng:
    python extract_contract_fields.py input/20260106212919160.md
    python extract_contract_fields.py input/*.md --model <ten_model>
    python extract_contract_fields.py input/foo.md --no-llm        # chỉ regex
    python extract_contract_fields.py input/foo.md -o out.json      # chỉ định file ra
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ====================================================================
# CONFIG - SỬA Ở ĐÂY ĐỂ CHẠY TRỰC TIẾP (python extract_contract_fields.py)
# ====================================================================
# Danh sách file .md cần xử lý khi không truyền tham số CLI.
DEFAULT_INPUTS = [
    "input/20260106212919160.md",
]

# Tên model LLM. Có thể để trống "" nếu server tự chọn model mặc định.
MODEL_NAME = os.getenv("MODEL_NAME_CORE", "")

# Địa chỉ API LLM.
LLM_API_URL = os.getenv("LLM_API_URL_CORE", "http://10.0.99.116:8070/v1/chat/completions")

# Bật/tắt gọi LLM và chế độ stream.
USE_LLM = True
STREAM = False
MAX_TOKEN = 40000
LLM_MAX_MARKDOWN_CHARS = 32000  # giới hạn markdown gửi LLM (0 = không rút gọn)


# ================== SCHEMA ==================

def empty_schema():
    """Trả về một bản schema rỗng (tất cả null) để đảm bảo output luôn đủ trường."""
    return {
        "so_hop_dong": None,
        "loai_hop_dong": None,
        "thoi_han_hop_dong": None,
        "ngay_hieu_luc": None,
        "ngay_ket_thuc": None,
        "chuc_vu": None,
        "vi_tri_cong_viec": None,
        "phong_ban": None,
        "quan_ly_truc_tiep": None,
        "dia_chi_thuong_tru": None,
        "dia_chi_hien_tai": None,
        "cccd": {"so": None, "ngay_cap": None, "noi_cap": None},
        "ma_so_bhxh": None,
        "mst_ca_nhan": None,
        "luong_co_ban": None,
        "phu_cap": {
            "xang_xe": None,
            "dien_thoai": None,
            "an_trua": None,
            "trach_nhiem": None,
            "hqcv": None,
        },
        "thuong_nang_luc": None,
        "tong_thu_nhap": None,
    }


# ================== LLM CLIENT ==================

def call_chat_api(messages,
                  model=None,
                  stream=False,
                  tools=None,
                  max_token=MAX_TOKEN,
                  host=None,
                  resonning=False):
    """Gọi API chat (giữ nguyên chữ ký/hành vi hàm có sẵn của bạn).

    Trả về object `requests.Response` thô. Mặc định ở đây để stream=False cho dễ parse.
    """
    payload = {
        "model": model or MODEL_NAME,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 64,
            "num_ctx": max_token,
        },
        "tools": tools,
        "chat_template_kwargs": {"enable_thinking": resonning},
    }
    response = requests.post(host or LLM_API_URL, json=payload, stream=stream, timeout=300)
    return response


def _extract_text_from_payload(data):
    """Lấy nội dung text từ 1 dict response, hỗ trợ cả format OpenAI lẫn Ollama."""
    if not isinstance(data, dict):
        return ""
    # OpenAI: {"choices": [{"message": {"content": ...}}]} hoặc delta khi stream
    choices = data.get("choices")
    if choices:
        ch = choices[0]
        msg = ch.get("message") or ch.get("delta") or {}
        return msg.get("content") or ""
    # Ollama: {"message": {"content": ...}}
    msg = data.get("message")
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return ""


def read_llm_content(response, stream):
    """Đọc toàn bộ text trả về từ response (xử lý cả stream SSE/JSON-lines và non-stream)."""
    if not stream:
        try:
            return _extract_text_from_payload(response.json())
        except ValueError:
            return response.text or ""

    parts = []
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if line.startswith("data:"):  # SSE (OpenAI)
            line = line[len("data:"):].strip()
        if line in ("[DONE]", ""):
            continue
        try:
            parts.append(_extract_text_from_payload(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return "".join(parts)


def parse_json_object(text):
    """Tách object JSON đầu tiên ra khỏi text (model đôi khi bọc ```json ... ``` hoặc kèm chữ)."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("Không parse được JSON từ LLM output.")
        return None


# ================== LỚP REGEX (FIELD CỐ ĐỊNH) ==================

def _first(pattern, text, group=1, flags=re.IGNORECASE):
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None


def _norm_money(raw):
    """Chuẩn hoá chuỗi tiền: '5.007.600 VNĐ/tháng' -> '5.007.600 VNĐ/tháng' (gọn khoảng trắng)."""
    if not raw:
        return None
    return re.sub(r"\s+", " ", raw).strip()


def _extract_loai_hop_dong(text):
    """Tiêu đề hợp đồng chính ở đầu văn bản (không lấy PHỤ LỤC, không suy từ thời hạn)."""
    for m in re.finditer(
        r"^#{1,6}\s*(HỢP\s*ĐỒNG[^\n#]+?)\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    ):
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        if re.search(r"phụ\s*lục", title, re.IGNORECASE):
            continue
        return title
    for m in re.finditer(
        r"^(HỢP\s*ĐỒNG\s+(?:LAO\s*ĐỘNG|THỬ\s*VIỆC|ĐÀO\s*TẠO))\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    ):
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return None


def regex_extract(text):
    """Trích các trường có format ổn định bằng regex (neo theo nhãn để tránh nhầm)."""
    out = {}

    # Số hợp đồng: ưu tiên HĐLĐ (hợp đồng chính), không lấy PLHĐLĐ (phụ lục)
    out["so_hop_dong"] = _first(r"(\d+/\d+/H[ĐĐ]L[ĐĐ][-\w]*)", text)

    loai = _extract_loai_hop_dong(text)
    if loai:
        out["loai_hop_dong"] = loai

    # Thời hạn & ngày
    out["thoi_han_hop_dong"] = _first(r"Thời hạn của hợp đồng:\s*(\d+\s*tháng)", text)
    m = re.search(r"ngày\s*(\d{2}/\d{2}/\d{4})\s*đến ngày\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    if m:
        out["ngay_hieu_luc"] = m.group(1)
        out["ngay_ket_thuc"] = m.group(2)
    # Ngày hiệu lực dạng chữ: "có hiệu lực kể từ ngày 06 tháng 06 năm 2025"
    eff = re.search(r"hiệu lực kể từ ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", text, re.IGNORECASE)
    if eff and not out.get("ngay_hieu_luc"):
        out["ngay_hieu_luc"] = f"{int(eff.group(1)):02d}/{int(eff.group(2)):02d}/{eff.group(3)}"

    # CCCD: số 12 chữ số + ngày cấp + nơi cấp
    cccd = {"so": None, "ngay_cap": None, "noi_cap": None}
    cccd["so"] = _first(r"(?:CCCD|Căn cước|Hộ chiếu)[^\d]{0,40}(\d{12})", text)
    cccd["ngay_cap"] = _first(r"Cấp ngày:\s*(\d{2}/\d{2}/\d{4})", text)
    cccd["noi_cap"] = _first(r"Nơi cấp:\s*([^\n]+?)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if any(cccd.values()):
        out["cccd"] = cccd

    # Mã số BHXH (thường 10 chữ số) và MST cá nhân
    out["ma_so_bhxh"] = _first(r"Mã số BHXH:\s*(\d{8,12})", text)
    out["mst_ca_nhan"] = _first(r"Mã số thuế cá nhân:\s*(\d{10,13})", text)

    # Địa chỉ
    out["dia_chi_thuong_tru"] = _first(r"Địa chỉ thường trú:\s*([^\n]+)", text)
    out["dia_chi_hien_tai"] = _first(r"Địa chỉ hiện tại:\s*([^\n]+)", text)

    # Tiền (fallback nếu LLM không lấy được)
    out["luong_co_ban"] = _norm_money(_first(r"Lương cơ bản\s*:?\s*([\d\.]+\s*VN[ĐD][^\n]*)", text))
    out["thuong_nang_luc"] = _norm_money(
        _first(r"Thưởng\s*năng\s*lực\s*:?\s*([^\n]+)", text)
        or _first(r"Thưởng\s*NL\s*:?\s*([^\n]+)", text)
    )
    out["tong_thu_nhap"] = _norm_money(_first(r"Tổng thu nhập:\s*([\d\.]+\s*VN[ĐD][^\n]*)", text))

    return {k: v for k, v in out.items() if v}


# ================== LỚP LLM (FIELD NGỮ NGHĨA) ==================

SYSTEM_PROMPT = (
    "Bạn là trợ lý trích xuất thông tin từ hợp đồng lao động tiếng Việt. "
    "Chỉ trả về MỘT object JSON hợp lệ, không kèm giải thích, không markdown."
)

USER_PROMPT_TEMPLATE = """Trích xuất thông tin từ nội dung hợp đồng dưới đây và trả về ĐÚNG schema JSON sau.

Quy tắc bắt buộc:
- Trường nào không tìm thấy trong văn bản thì để null. TUYỆT ĐỐI không bịa.
- "mst_ca_nhan" là mã số thuế của NGƯỜI LAO ĐỘNG (cá nhân), KHÔNG lấy mã số thuế của Công ty.
- "vi_tri_cong_viec" = chức danh công việc của người lao động; "chuc_vu" có thể giống vi_tri_cong_viec.
- "loai_hop_dong": lấy ĐÚNG tiêu đề hợp đồng ở đầu văn bản (vd "### HỢP ĐỒNG LAO ĐỘNG", "HỢP ĐỒNG THỬ VIỆC").
  KHÔNG tự thêm "có thời hạn", "xác định thời hạn", "không xác định thời hạn" chỉ vì Điều 1 có "Thời hạn của hợp đồng: X tháng"
  — số tháng thuộc trường "thoi_han_hop_dong", không gộp vào loại hợp đồng.
  Nếu tiêu đề chỉ ghi "HỢP ĐỒNG LAO ĐỘNG" thì trả đúng như vậy, không đổi thành "HỢP ĐỒNG LAO ĐỘNG có thời hạn".
  Bỏ qua tiêu đề "PHỤ LỤC HỢP ĐỒNG ...".
- Thông tin lương/phụ cấp/thưởng năng lực/tổng thu nhập thường nằm ở phần PHỤ LỤC, hãy đọc hết văn bản.
- "thuong_nang_luc": thưởng năng lực (có thể ghi "Thưởng năng lực", "Thưởng NL"); lấy số tiền hoặc mô tả đúng như văn bản.
- Nếu một dòng phụ cấp gộp nhiều loại (vd "Phụ cấp xăng xe, điện thoại: 500.000"),
  hãy điền CÙNG giá trị đó cho từng loại liên quan (xang_xe và dien_thoai).
- "hqcv" = phụ cấp hiệu quả/hoàn thành công việc.
- Giữ nguyên định dạng số tiền như trong văn bản (vd "5.007.600 VNĐ/tháng").

Schema JSON:
{schema}

Nội dung hợp đồng:
\"\"\"
{content}
\"\"\"
"""


def llm_extract(text, model, stream=False):
    """Gọi LLM để trích toàn bộ schema; trả về dict (hoặc None nếu lỗi)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            schema=json.dumps(empty_schema(), ensure_ascii=False, indent=2),
            content=text,
        )},
    ]
    try:
        resp = call_chat_api(messages, model=model, stream=stream)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Gọi LLM thất bại: %s", exc)
        return None
    content = read_llm_content(resp, stream)
    return parse_json_object(content)


# ================== LLM SỬA CHÍNH TẢ OCR ==================

OCR_CORRECT_SYSTEM = (
    "Bạn là chuyên gia chỉnh sửa kết quả OCR tiếng Việt. "
    "Chỉ trả về văn bản đã sửa, không giải thích, không bọc markdown code fence."
)

OCR_CORRECT_USER_TEMPLATE = """Dưới đây là văn bản OCR từ hợp đồng lao động tiếng Việt — thường thiếu dấu, sai chính tả, dính chữ.

Nhiệm vụ:
1. Sửa dấu thanh / chính tả tiếng Việt cho đúng (vd: Nguyn→Nguyễn, Hà Ni→Hà Nội, HOP DÒNG→HỢP ĐỒNG, Giám đc→Giám đốc).
2. Tách khoảng trắng hợp lý khi OCR dính chữ (vd: 024.6666.5646www.aipt→024.6666.5646 www.aipt).
3. GIỮ nguyên cấu trúc dòng, bullet, heading markdown, số liệu, mã số, ngày tháng, email, URL.
4. KHÔNG thêm điều khoản mới, KHÔNG xóa nội dung, KHÔNG bịa thông tin không có trong bản OCR.
5. Nếu không chắc một từ, giữ nguyên từ OCR.

Văn bản OCR:
\"\"\"
{content}
\"\"\"
"""


def _strip_llm_text_wrapper(text):
    """Bỏ ``` / ```markdown nếu model vẫn bọc output."""
    if not text:
        return text
    text = text.strip()
    fence = re.match(r"^```(?:markdown|md|text)?\s*\n?(.*?)\n?```\s*$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def llm_correct_ocr(text, model=None, stream=False, max_chars=None):
    """Sửa dấu/chính tả OCR bằng LLM. Lỗi hoặc output xấu -> trả về text gốc."""
    if not text or not text.strip():
        return text
    slim = trim_markdown_for_llm(text, max_chars=max_chars)
    messages = [
        {"role": "system", "content": OCR_CORRECT_SYSTEM},
        {"role": "user", "content": OCR_CORRECT_USER_TEMPLATE.format(content=slim)},
    ]
    try:
        resp = call_chat_api(messages, model=model, stream=stream)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("LLM sửa OCR thất bại: %s -> giữ bản gốc.", exc)
        return text

    corrected = _strip_llm_text_wrapper(read_llm_content(resp, stream))
    if not corrected or not corrected.strip():
        logger.warning("LLM sửa OCR trả về rỗng -> giữ bản gốc.")
        return text
    # Bảo vệ: model đôi khi cắt quá nhiều
    if len(corrected) < max(80, int(len(slim) * 0.45)):
        logger.warning(
            "LLM sửa OCR ngắn bất thường (%d -> %d) -> giữ bản gốc.",
            len(slim), len(corrected),
        )
        return text
    logger.info("Đã sửa chính tả OCR bằng LLM (%d -> %d ký tự).", len(text), len(corrected))
    return corrected


# ================== MERGE ==================

# Các field (dạng dotted path) mà regex đáng tin hơn -> ghi đè LLM.
REGEX_OVERRIDE = {
    "so_hop_dong",
    "thoi_han_hop_dong",
    "ngay_hieu_luc",
    "ngay_ket_thuc",
    "cccd.so",
    "cccd.ngay_cap",
    "cccd.noi_cap",
    "ma_so_bhxh",
    "mst_ca_nhan",
    "dia_chi_thuong_tru",
    "dia_chi_hien_tai",
    "loai_hop_dong",
    "phong_ban",
}

# Field thường trống trên mẫu -> không bắt buộc gọi LLM vì thiếu các field này
OPTIONAL_FIELDS = frozenset({"phu_cap.an_trua"})

# Từ khóa giữ lại khi rút gọn markdown gửi LLM (phụ lục / lương)
_LLM_KEEP_KEYWORDS = re.compile(
    r"phụ\s*lục|phu\s*luc|lương|luong|phụ\s*cấp|phu\s*cap|thu\s*nhập|thưởng|thuong|năng\s*lực|PLHĐ|PLHD",
    re.IGNORECASE,
)


def _get(d, dotted):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _set(d, dotted, value):
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _iter_paths(schema, prefix=""):
    for key, val in schema.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            yield from _iter_paths(val, prefix=f"{path}.")
        else:
            yield path


def count_missing(data, ignore=OPTIONAL_FIELDS):
    """Đếm số field schema còn trống sau regex (bỏ qua field optional)."""
    n = 0
    for path in _iter_paths(empty_schema()):
        if path in ignore:
            continue
        if _get(data, path) in (None, ""):
            n += 1
    return n


def trim_markdown_for_llm(text, max_chars=None):
    """Rút gọn markdown gửi LLM: giữ trang đầu/cuối + đoạn có lương/phụ lục (không đụng OCR)."""
    if max_chars is None:
        max_chars = LLM_MAX_MARKDOWN_CHARS
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    chunks = re.split(r"(?=\n## Page \d+)", text)
    if len(chunks) <= 1:
        return text[:max_chars]
    kept = []
    for i, chunk in enumerate(chunks):
        if i < 3 or i >= len(chunks) - 2 or _LLM_KEEP_KEYWORDS.search(chunk):
            kept.append(chunk)
    slim = "".join(kept)
    if len(slim) > max_chars:
        slim = slim[:max_chars]
    logger.info("Rút gọn markdown LLM: %d -> %d ký tự", len(text), len(slim))
    return slim


def merge(llm_data, rx_data):
    """Gộp kết quả LLM + regex vào schema chuẩn."""
    result = empty_schema()
    llm_data = llm_data or {}

    # 1) Đổ kết quả LLM vào (chỉ lấy các path hợp lệ theo schema)
    for path in _iter_paths(result):
        val = _get(llm_data, path)
        if val not in (None, ""):
            _set(result, path, val)

    # 2) Áp regex
    for path in _iter_paths(result):
        rx_val = _get(rx_data, path)
        if rx_val in (None, ""):
            continue
        if path in REGEX_OVERRIDE or _get(result, path) in (None, ""):
            _set(result, path, rx_val)

    return result


# ================== PIPELINE ==================

def process_file(md_path, model, use_llm=True, stream=False):
    text = Path(md_path).read_text(encoding="utf-8")
    rx_data = regex_extract(text)
    llm_data = llm_extract(text, model, stream=stream) if use_llm else None
    if use_llm and llm_data is None:
        logger.warning("LLM không trả kết quả cho %s, dùng regex-only.", md_path)
    return merge(llm_data, rx_data)


def main():
    parser = argparse.ArgumentParser(description="Trích xuất thông tin hợp đồng từ file .md ra JSON.")
    parser.add_argument("inputs", nargs="*", default=DEFAULT_INPUTS,
                        help="Đường dẫn file .md (mặc định lấy từ DEFAULT_INPUTS trong code)")
    parser.add_argument("--model", default=MODEL_NAME, help="Tên model LLM (mặc định MODEL_NAME trong code/env)")
    parser.add_argument("--no-llm", action="store_true", help="Chỉ dùng regex, không gọi LLM")
    parser.add_argument("--stream", action="store_true", default=STREAM, help="Bật chế độ stream khi gọi LLM")
    parser.add_argument("-o", "--output", help="Ghi JSON ra file này (chỉ áp dụng khi có 1 input)")
    args = parser.parse_args()

    inputs = args.inputs or DEFAULT_INPUTS
    use_llm = USE_LLM and not args.no_llm

    results = {}
    for md_path in inputs:
        if not Path(md_path).is_file():
            logger.warning("Bỏ qua (không tìm thấy): %s", md_path)
            continue
        logger.info("Đang xử lý: %s", md_path)
        data = process_file(md_path, args.model, use_llm=use_llm, stream=args.stream)
        results[md_path] = data

        # Ghi file .fields.json cạnh file gốc
        out_path = args.output if (args.output and len(inputs) == 1) \
            else str(Path(md_path).with_suffix(".fields.json"))
        Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Đã lưu: %s", out_path)

    # In ra stdout cho tiện kiểm tra
    payload = next(iter(results.values())) if len(results) == 1 else results
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
