import os
import re
import fitz  # PyMuPDF
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownNodeParser

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
# ĐIỀN API KEY CỦA BẠN VÀO ĐÂY (Lấy từ cloud.llamaindex.ai)
LLAMA_CLOUD_API_KEY = "llx-tYVrs39ypcwv26BoLkGzKVFWHIDTRRxIThhkZb9TjjetPkH5" 
FILE_PATH = r"D:\RAG_MODEL\52500174(report_2)(2).pdf"
BATCH_SIZE = 10 # Với LlamaParse (Cloud), bạn có thể xử lý nhiều trang hơn mỗi đợt

# Thiết lập biến môi trường để LlamaParse có thể nhận dạng
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY

# ==========================================
# 1. HÀM CẮT PDF (Tuỳ chọn, nhưng nên dùng cho file lớn)
# ==========================================
def split_pdf_into_batches(input_pdf_path, output_folder="pdf_batches", pages_per_batch=10):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(input_pdf_path)
    total_pages = len(doc)
    batch_files = []
    
    print(f"✂️ Đang cắt PDF ({total_pages} trang) thành các file ({pages_per_batch} trang/file)...")
    for i in range(0, total_pages, pages_per_batch):
        new_doc = fitz.open()
        start_page = i
        end_page = min(i + pages_per_batch - 1, total_pages - 1)
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
        
        batch_filename = os.path.join(output_folder, f"batch_{start_page}_to_{end_page}.pdf")
        new_doc.save(batch_filename)
        new_doc.close()
        batch_files.append(batch_filename)
        
    doc.close()
    return batch_files

# ==========================================
# 2. KHỞI TẠO LLAMAPARSE (Cấu hình chuẩn cho Báo cáo/Toán học)
# ==========================================
def setup_llamaparse():
    parser = LlamaParse(
        result_type="markdown", 
        verbose=True,
        language="vi",          
        num_workers=4,          
        # BẬT DÒNG NÀY LÊN ĐỂ XỬ LÝ TOÁN VÀ BẢNG BIỂU PHỨC TẠP NHẤT
        premium_mode=True, 
    )
    return parser

# ==========================================
# 3. LÀM SẠCH VÀ CHUNKING CHUYÊN NGHIỆP
# ==========================================
def clean_markdown_with_regex(text):
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) 
    # Nếu file có toán học LaTeX, cần thận trọng khi làm sạch các ký tự đặc biệt.
    # Dòng dưới đây có thể bị lược bớt để giữ nguyên mã LaTeX.
    # text = re.sub(r'[^\w\s#*.|,\-\:\/=()]', '', text).strip() 
    return text

def advanced_chunking(markdown_text):
    """
    Sử dụng công cụ chia nhỏ theo cấu trúc Markdown của LlamaIndex.
    Đây là phương pháp ưu việt hơn Regex thuần túy, an toàn tuyệt đối với công thức Toán.
    """
    parser = MarkdownNodeParser()
    
    # LlamaIndex cần định dạng Node
    from llama_index.core.schema import Document
    doc = Document(text=markdown_text)
    
    # Thực hiện chia nhỏ
    nodes = parser.get_nodes_from_documents([doc])
    
    # Trích xuất văn bản từ các Node
    chunks = [node.text for node in nodes]
    return chunks

# ==========================================
# LUỒNG ĐIỀU PHỐI CHÍNH
# ==========================================
if __name__ == "__main__":
    all_final_chunks = []
    
    try:
        # BƯỚC 1: Cắt PDF (Nếu máy bạn khỏe hoặc file không quá lớn, có thể bỏ qua bước này và gửi thẳng file gốc)
        batch_files = split_pdf_into_batches(FILE_PATH, pages_per_batch=BATCH_SIZE)
        
        # BƯỚC 2: Khởi tạo Parser
        parser = setup_llamaparse()
        
        # BƯỚC 3: Xử lý từng file qua Cloud
        for batch_index, pdf_batch in enumerate(batch_files):
            print(f"\n🚀 ĐANG GỬI BATCH {batch_index + 1}/{len(batch_files)} LÊN LLAMAPARSE CLOUD...")
            
            # LlamaParse sẽ tự động đọc, hiểu cấu trúc, bảng biểu và trả về Markdown
            documents = parser.load_data(pdf_batch)
            
            # Thường trả về 1 Document cho 1 file PDF
            md_batch = documents[0].text 

            # BƯỚC 4: Làm sạch và Chunking
            clean_md = clean_markdown_with_regex(md_batch)
            
            # Sử dụng bộ chia nhỏ Markdown chuyên nghiệp
            chunks = advanced_chunking(clean_md)
            
            all_final_chunks.extend(chunks)
            print(f"   ✔️ Hoàn tất Batch {batch_index + 1}. Thu được {len(chunks)} chunks.")

        # BƯỚC 5: TỔNG KẾT
        print("\n" + "🌟" * 25)
        print(f"🎉 HOÀN THÀNH QUY TRÌNH VỚI LLAMAPARSE!")
        print(f"📊 Tổng số lượng chunks: {len(all_final_chunks)}")
        print("🌟" * 25)
        
        print("\n" + "="*40)
        a = 1
        for i in all_final_chunks:
            print(f"chunk {a}")
            print(i)
            a+=1

    except Exception as e:
        print(f"\n❌ Lỗi hệ thống: {e}")