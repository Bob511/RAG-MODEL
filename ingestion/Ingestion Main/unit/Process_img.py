import os
from dotenv import load_dotenv
from PIL import Image
import textwrap
import time
import re
from google import genai
from google.genai import types


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def describe_image_with_gemini(image_path: str) -> str:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)
    """Đọc ảnh từ ổ cứng, gửi lên Gemini với Mega-Prompt chuyên dụng."""
    filename = os.path.basename(image_path)
    try:
        img = Image.open(image_path)
        
        # Mega-Prompt an toàn của Nam
        prompt = textwrap.dedent(f"""
            Bạn là một chuyên gia Data Engineer và Chuyên viên Xử lý Dữ liệu RAG. Nhiệm vụ của bạn là phân tích hình ảnh đính kèm và chuyển đổi nó thành định dạng văn bản có cấu trúc tốt nhất cho hệ thống lưu trữ Vector.

            TRƯỚC TIÊN, hãy tự xác định xem hình ảnh này thuộc loại nào và áp dụng QUY TẮC TƯƠNG ỨNG dưới đây:

            1. NẾU LÀ BIỂU ĐỒ HOẶC ĐỒ THỊ (Chart/Graph):
               - Trích xuất toàn bộ số liệu thành MỘT bảng Markdown.
               - Cột đầu tiên là trục X, các cột tiếp theo là trục Y.
               - Nếu dữ liệu không nằm đúng vạch, hãy ước lượng và thêm dấu `~`.
               - Viết kèm một dòng Header (H3) tóm tắt: `### Dữ liệu biểu đồ: [Chủ đề chính]`

            2. NẾU LÀ CÔNG THỨC TOÁN HỌC/HÓA HỌC (Formula/Equation):
               - Chuyển đổi chính xác sang định dạng LaTeX.
               - Bắt buộc bọc công thức trong cặp dấu `$$`.

            3. NẾU LÀ SƠ ĐỒ HOẶC HÌNH MINH HỌA (Diagram/Illustration):
               - Viết một đoạn văn Markdown chi tiết (2-3 câu) mô tả logic, cấu trúc hoặc luồng quy trình.
            4. Nếu là một ảnh chứa text thuần (Text Image):
               - Trích xuất toàn bộ văn bản và trình bày dưới dạng Markdown, giữ nguyên cấu trúc dòng và đoạn. KHÔNG THÊM BỚT VĂN BẢN NÀO KHÁC

            RÀNG BUỘC TỐI THƯỢNG:
            - Bắt đầu kết quả của bạn bằng dòng: `> [Ngữ cảnh: Hình ảnh được trích xuất từ {filename}]`
            - KHÔNG giải thích quá trình làm việc. CHỈ TRẢ VỀ KẾT QUẢ ĐÃ FORMAT.
        """).strip()

        response = client.models.generate_content(
            model="gemini-2.5-flash", # Đã cập nhật lên bản mới nhất
            contents=[prompt, img],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        return response.text
    except Exception as e:
        print(f"    [!] Lỗi Gemini tại {filename}: {e}")
        return f"> [Lỗi trích xuất ảnh {filename}]\n"

def process_images_in_text(text: str, output_img_dir: str) -> str:
    """Tìm thẻ ảnh Markdown và thay thế bằng nội dung phân tích từ Gemini."""
    print("2.2. Bắt đầu quét văn bản và phân tích hình ảnh bằng LLM...")
    # Pattern tìm ![caption](path)
    pattern = r'!\[.*?\]\((.*?)\)'

    def replacer(match):
        img_filename = match.group(1)
        # Marker thường chỉ để tên file, ta cần nối với đường dẫn thư mục output
        full_img_path = os.path.join(output_img_dir, img_filename)
        
        if os.path.exists(full_img_path):
            print(f"   -> Đang xử lý: {img_filename}")
            analysis = describe_image_with_gemini(full_img_path)
            time.sleep(4) # Chống lỗi 429
        return f"\n\n{analysis}\n\n"
        

    return re.sub(pattern, replacer, text)