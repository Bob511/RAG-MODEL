import os
import time
import uuid
import shutil
import glob
import re
import textwrap
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
import ollama
import chromadb
from langchain_core.documents import Document
# Marker & Langchain
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Cấu hình Gemini Client (SDK v2)
ChromaAPI = os.getenv("ChromaAPI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)





def get_all_pdf_files(folder_path):
    # Sử dụng glob để lọc ra các file có đuôi .pdf
    # recursive=True giúp tìm cả trong các thư mục con nếu cần
    query = os.path.join(folder_path, "*.pdf")
    return glob.glob(query)
# ==========================================
# BƯỚC 1: SETTING CONVERTER
# ==========================================
def setup_converter():
    """Khởi động Engine AI trên GPU và cấu hình thư mục lưu trữ."""
    print("1. Đang khởi động Engine AI trên ổ D...")
    os.environ["HF_HOME"] = "D:/Marker_Cache"
    return load_all_models()

# ==========================================
# BƯỚC 2: HÀM XỬ LÝ ẢNH VỚI MEGA-PROMPT
# ==========================================
def describe_image_with_gemini(image_path: str) -> str:
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

# ==========================================
# BƯỚC 3: CLEANING
# ==========================================
def clean_markdown(raw_text: str, image_dict) -> str:
    """Dọn dẹp định dạng rác."""
    print("3. Đang dọn dẹp (Cleaning) mã Markdown...")
    for name in image_dict.keys():
        raw_text = raw_text.replace(f"![Ngữ cảnh: Hình ảnh được trích xuất từ {name}]", "") # Loại bỏ thẻ ảnh cũ nếu còn sót
        raw_text = raw_text.replace(f"![Lỗi trích xuất ảnh {name}]", "")
    text = re.sub(r'\n{3,}', '\n\n', raw_text)
    return text.strip()

# ==========================================
# BƯỚC 4: RECURSIVE CHUNKING
# ==========================================
def chunks_for_rag(text):
    """
    Hàm chunking tối ưu:
    1. Ưu tiên giữ các bảng nguyên vẹn.
    2. Phân tách theo Header (##).
    3. Giao các đoạn văn bản dài (>chunk_size) cho Langchain xử lý.
    """
    final_chunks = []
    
    # BƯỚC 1: XÓA CÁC THẺ ẢNH RÁC CHEN NGANG BẢNG
    # Điều này ngăn chặn việc bảng bị đứt gãy do text = re.sub(r'\n\n', '\n', text)
    
    # BƯỚC 2: PHÂN TÁCH THEO HEADER
    # Chia nhỏ văn bản tại mỗi thẻ '## ', giữ lại thẻ đó ở đầu mỗi phần
    sections = re.split(r'(?=\n#{1,6} )', "\n" + text)
    
    # Khởi tạo Text Splitter của Langchain cho những khối văn bản dài KHÔNG PHẢI là bảng
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    # BƯỚC 3: DUYỆT TỪNG SECTION
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Kiểm tra xem section này có chứa đặc điểm của Bảng Markdown hay không (chứa |---|)
        has_table = re.search(r'\|[\s\-:]+\|', section)
        
        if has_table:
            # NẾU CÓ BẢNG: Ép giữ nguyên toàn bộ khối này thành 1 chunk duy nhất
            # Chấp nhận dung lượng chunk > 1000 để bảo toàn ngữ nghĩa của bảng
            final_chunks.append(Document(page_content=section))
        else:
            # NẾU KHÔNG CÓ BẢNG:
            if len(section) <= 2000:
                # Nếu ngắn, lưu luôn
                final_chunks.append(Document(page_content=section))
            else:
                # Nếu quá dài, giao cho Langchain tiếp tục băm nhỏ ra
                sub_chunks = text_splitter.create_documents([section])
                final_chunks.extend(sub_chunks)
                
    return final_chunks

def save_chunks_to_chromadb_cloud(chunks):
    """
    Nhúng các chunks bằng Ollama và đẩy thẳng lên ChromaDB Cloud.
    """
    print("☁️ Đang kết nối tới ChromaDB Cloud...")
    
    # 1. Khởi tạo CloudClient (Thay thế cho PersistentClient)
    client = chromadb.CloudClient(
    api_key= ChromaAPI,
    tenant='4798bb4f-8541-44e6-ab6f-6b6594fcef7a',
    database='BIZRAG'
    )
    COLLECTION_NAME = 'Bigchild'
    # 2. Tạo hoặc lấy Collection trên Cloud
    collection = client.get_or_create_collection(name = COLLECTION_NAME)
    
    documents = []
    metadatas = []
    ids = []
    embeddings = []

    print(f"🔄 Bắt đầu embedding và chuẩn bị dữ liệu cho {len(chunks)} chunks...")
    
    # 3. Duyệt qua từng chunk để lấy vector
    for i, chunk in enumerate(chunks):
        # Gọi Ollama chạy local để lấy vector
        response = ollama.embed(
            model='qwen3-embedding:0.6b', 
            input=chunk.page_content
        )
        vector = response['embeddings'][0]
        
        documents.append(chunk.page_content)
        
        meta = chunk.metadata if chunk.metadata else {}
        meta["chunk_id"] = i
        metadatas.append(meta)
        
        ids.append(f"chunk_{i}")
        embeddings.append(vector)
        
        print(f"  -> Đã nhúng xong chunk {i+1}/{len(chunks)}")

    # 4. Đẩy toàn bộ lên Cloud
    if documents:
        print("🚀 Đang tải dữ liệu lên ChromaDB Cloud... Vui lòng đợi.")
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print("\n✅ ĐÃ LƯU THÀNH CÔNG LÊN CHROMA CLOUD!")
    else:
        print("\n⚠️ Không có dữ liệu để lưu.")
        
    return collection
# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        input_folder = input("Nhập tên foler chứa file PDF: ").strip()
        if not os.path.exists(input_folder):
            print(f"Thư mục '{input_folder}' không tồn tại. Vui lòng tạo thư mục và thêm file PDF vào đó.")
            exit(1)
        else:
            pdf_files = get_all_pdf_files(input_folder)
        if not pdf_files:
            print(f"Không tìm thấy file PDF nào trong thư mục '{input_folder}'. Vui lòng thêm file PDF vào đó.")
            exit(1)
        else:
            file_name = input("Nhập tên file: ").strip() 
            PDF_FILE = os.path.join(input_folder, file_name)
            print(f"Đã tìm thấy file PDF: {PDF_FILE}")
        IMG_DIR = "extracted_images"
        FINAL_MD = "final_output.md"

        # 1. Setup
        model_lst = setup_converter()

        # 2. Convert & Save Images
        print(f"2.1. Marker đang bóc tách PDF...")
        full_text, images_dict, out_meta = convert_single_pdf(PDF_FILE, model_lst)
    
        # Lưu ảnh vật lý để hàm regex có thể đọc được
        os.makedirs(IMG_DIR, exist_ok=True)
        for name, img in images_dict.items():
            img.save(os.path.join(IMG_DIR, name))

        # 3. LLM Processing (Thay thế ảnh bằng Text)
        full_text_with_ai = process_images_in_text(full_text, IMG_DIR)

        # 4. Clean & Chunk
        final_text = clean_markdown(full_text_with_ai, images_dict)
        chunks = chunks_for_rag(final_text)

        # 5. Export
        with open(FINAL_MD, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"---chunk {i}---")
                f.write(f"\n{chunk}\n\n")
        final_vector_db = []
        collection = save_chunks_to_chromadb_cloud(chunks)
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
    finally:
        if os.path.exists(IMG_DIR):
            shutil.rmtree(IMG_DIR)