
import ollama
import os
import re
import fitz  # PyMuPDF
import gc
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb
from dotenv import load_dotenv


load_dotenv()
FILE_PATH = r"D:\ragmodel\52500174(Report_2)(2).pdf"
MODEL_VISION = "qwen2.5vl:3b"
os.getenv("ChromaAPI")

def setup_docling():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.picture_description_options= True
    pipeline_options.do_formula_enrichment = False
    pipeline_options.ocr_options = EasyOcrOptions(lang=["en", "vi"])
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
def split_pdf_into_batches(input_path, pages_per_batch=5):
    """Cắt file PDF lớn thành các file nhỏ."""
    output_folder = "pdf_batches"
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(input_path)
    batch_files = []
    
    print(f"✂️ Đang cắt PDF ({len(doc)} trang) thành các batch {pages_per_batch} trang...")
    for i in range(0, len(doc), pages_per_batch):
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=min(i + pages_per_batch - 1, len(doc) - 1))
        batch_path = os.path.join(output_folder, f"batch_{i}.pdf")
        new_doc.save(batch_path)
        new_doc.close()
        batch_files.append(batch_path)
    doc.close()
    return batch_files


def ask_qwen_vision(image_path, content_type):
    """Sử dụng Qwen để xử lý linh hoạt tùy theo loại nội dung"""
    if content_type == "math":
        prompt = "Đây là ma trận hoặc công thức toán. Chuyển nó thành mã LaTeX chuẩn, kẹp trong $$. Chỉ trả về mã."
        prefix = ""
    elif content_type == "table":
        prompt = "Chuyển bảng này thành định dạng Markdown Table chuẩn."
        prefix = "\n"
    else: # picture/chart
        prompt = "Phân tích nội dung biểu đồ/hình ảnh này chi tiết bằng tiếng Việt."
        prefix = "\n> 📊 **[Phân tích từ Vision AI]:** "
        
    try:
        response = ollama.generate(model=MODEL_VISION, prompt=prompt, images=[image_path])
        if os.path.exists(image_path):
            os.remove(image_path)
        return f"\n{prefix}{response['response'].strip()}\n\n"
    except Exception as e:
        return f"\n[Lỗi Vision AI: {e}]\n"


def extract_and_process_images_in_batch(pdf_batch_path, img_folder="extracted_images"):
    """
    Quét PDF, trích xuất ảnh thật (bỏ qua icon/logo nhỏ), lưu vào folder tạm,
    gửi cho Qwen Vision, XÓA ẢNH, và trả về một chuỗi Markdown chứa mô tả.
    """
    # 1. Tạo folder chứa ảnh tạm nếu chưa tồn tại
    os.makedirs(img_folder, exist_ok=True)
    
    doc = fitz.open(pdf_batch_path)
    image_markdown_results = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Lấy danh sách toàn bộ ảnh được nhúng trong trang
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            width = base_image["width"]
            height = base_image["height"]
            
            # TỐI ƯU: Lọc ảnh rác (icon, logo trang trí, đường gạch ngang)
            if width < 150 or height < 150:
                continue
                
            # 2. Lưu ảnh tạm vào FOLDER RIÊNG thay vì thư mục gốc
            img_filename = f"temp_img_p{page_num}_{img_index}.{ext}"
            img_path = os.path.join(img_folder, img_filename)
            
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            
            print(f"   📸 Đang phân tích ảnh ở Trang {page_num + 1} bằng Qwen Vision...")
            
            try:
                # 3. Đưa lên LLM thông qua hàm của bạn
                description = ask_qwen_vision(img_path, content_type="picture")
                
                # Lưu lại đoạn Markdown mô tả ảnh
                image_markdown_results.append(f"## 🖼️ Hình ảnh tại Trang {page_num + 1} (Mã {img_index})\n{description}")
            
            finally:
                # 4. BẢO ĐẢM XÓA ẢNH SAU KHI DÙNG
                # Đặt trong finally để dù ask_qwen_vision có bị lỗi (Exception) thì ảnh vẫn bị xóa
                if os.path.exists(img_path):
                    os.remove(img_path)
            
    doc.close()
    
    # 5. DỌN DẸP FOLDER: Xóa luôn folder tạm nếu bên trong không còn file nào
    if os.path.exists(img_folder) and not os.listdir(img_folder):
        try:
            os.rmdir(img_folder)
        except OSError:
            pass # Bỏ qua nếu folder không trống vì lý do nào đó
    
    # Nối tất cả các mô tả lại với nhau
    return "\n\n".join(image_markdown_results)

def chunks_from_pdf(text, chunk_size=1000, chunk_overlap=100):
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
    sections = re.split(r'(?=\n## )', "\n" + text)
    
    # Khởi tạo Text Splitter của Langchain cho những khối văn bản dài KHÔNG PHẢI là bảng
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
            if len(section) <= chunk_size:
                # Nếu ngắn, lưu luôn
                final_chunks.append(Document(page_content=section))
            else:
                # Nếu quá dài, giao cho Langchain tiếp tục băm nhỏ ra
                sub_chunks = text_splitter.create_documents([section])
                final_chunks.extend(sub_chunks)
                
    return final_chunks

def clean_markdown_with_regex(text):
    # CHỈ xóa các đường link rác (ví dụ: [image_1](...)). 
    # TUYỆT ĐỐI KHÔNG dùng [^\w\s...] để tránh xóa mất dấu \ của LaTeX
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) 
    return text.strip()
  
# ... (các import cũ của bạn) ...

# ==========================================
# CẤU HÌNH CHROMA CLOUD
# =========================================

def save_chunks_to_chromadb_cloud(chunks):
    """
    Nhúng các chunks bằng Ollama và đẩy thẳng lên ChromaDB Cloud.
    """
    print("☁️ Đang kết nối tới ChromaDB Cloud...")
    
    # 1. Khởi tạo CloudClient (Thay thế cho PersistentClient)
    client = chromadb.CloudClient(
    api_key='ck-Hvdk3rDbPi7uMeHtdpxSjJ1tBCvaAqmXkDAEnZJoVy2X',
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

if __name__ == "__main__":
    print("🚀 Khởi tạo hệ thống...")
    
    # 1. Khởi tạo công cụ Docling
    converter = setup_docling()
    
    # 2. Cắt PDF lớn thành các batch nhỏ và lưu vào folder
    batch_files = split_pdf_into_batches(FILE_PATH, pages_per_batch=5)
    
    # Danh sách tổng chứa kết quả cuối cùng
    all_final_chunks = []
    
    print("\n" + "="*30)
    print("⚙️ BẮT ĐẦU XỬ LÝ TỪNG BATCH")
    print("="*30)
    
    # 3. Duyệt qua từng file batch trong folder
    for batch_file in batch_files:
        print(f"\n📄 Đang xử lý: {batch_file}")
        
        try:
            # 1. Chuyển Text & Bảng sang Markdown bằng Docling
            result = converter.convert(batch_file)
            md_output = result.document.export_to_markdown()
            
            # 2. Xử lý chuyên sâu Hình Ảnh (Trích xuất -> Vision -> Chữ)
            image_descriptions = extract_and_process_images_in_batch(batch_file)
            
            # 3. CHÈN LẠI VÀO VĂN BẢN (Ghép nối)
            # Nếu có ảnh được phân tích, nối nó vào cuối nội dung Markdown của Batch này
            if image_descriptions.strip():
                md_output = md_output + "\n\n" + image_descriptions
            
            # 4. Dọn dẹp Regex và Chunking
            cleaned_text = clean_markdown_with_regex(md_output)
            chunks = chunks_from_pdf(cleaned_text)
            
            # Đưa vào danh sách tổng
            all_final_chunks.extend(chunks)
            print(f"  ✅ Trích xuất thành công {len(chunks)} chunks.")
            
        except Exception as e:
            print(f"  ❌ Lỗi khi xử lý {batch_file}: {e}")
            
        finally:
            # Dọn dẹp RAM và XÓA FILE BATCH để nhẹ máy
            if os.path.exists(batch_file):
                os.remove(batch_file)
            gc.collect()
    for i, doc in enumerate(all_final_chunks):
        print(f"\n--- chunk {i+1} ---")
        print(doc)
    print("\n" + "="*20 + " BẮT ĐẦU ĐƯA VÀO CƠ SỞ DỮ LIỆU " + "="*20)
    collection = save_chunks_to_chromadb_cloud(all_final_chunks)
    
   