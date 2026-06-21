import os
import re

def validate_and_cleanup_garbage(md_files):

    print(f"-> Đang quét kiểm tra file: {md_files}")
    content = md_files.read_text(encoding="utf-8")
    # Điều kiện 1: File quá ngắn hoặc rỗng (Parser lỗi)
    if len(content.strip()) < 10:
        print(f"   [CẢNH BÁO] File {md_files} quá ngắn ({len(content)} ký tự). Tiến hành xóa rác.")
        os.remove(md_files)
        return False
        # Điều kiện 2: Kiểm tra tỷ lệ ký tự lạ / ký tự nhiễu OCR phá hỏng Attention
        # Đếm các ký tự không phải chữ, số, dấu câu thông thường hoặc thẻ HTML/Markdown
    total_chars = len(content)
    garbage_chars = len(re.findall(r'[^a-zA-Z0-9\s\.,\?\!\:\;\(\)\[\]\{\}\+\-\*\/\\=\<\>\|\#]', content))
    garbage_ratio = garbage_chars / total_chars
        
    if garbage_ratio > 0.15: # Nếu rác chiếm > 15% tổng văn bản
        print(f"   [THẤT BẠI] Tỷ lệ rác quá cao ({garbage_ratio:.2%}). Xóa file để bảo vệ mô hình.")
        os.remove(md_files)
        return False
    if content.count("<table") != content.count("</table>"):
        print(f"   [THẤT BẠI] Phát hiện bảng HTML bị vỡ cấu trúc (mất thẻ đóng/mở). Tiến hành cô lập và xóa.")
        os.remove(md_files)
        return False
    print(f"   [VƯỢT QUA] File {md_files} đạt tiêu chuẩn chất lượng dữ liệu sạch.")
        
    print("--- [DONE] KẾT THÚC QUY TRÌNH VALIDATION ---")
    return True

if __name__ == "__main__":
    validate_and_cleanup_garbage()