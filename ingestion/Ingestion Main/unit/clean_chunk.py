import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_markdown(raw_text: str, image_dict: dict = None) -> str:
    """Dọn dẹp định dạng rác."""
    print("3. Đang dọn dẹp (Cleaning) mã Markdown...")
    text = raw_text
    if image_dict:
        for name in image_dict.keys():
            raw_text = raw_text.replace(f"![Ngữ cảnh|image|Hình ảnh: Hình ảnh được trích xuất từ {name}]", "") # Loại bỏ thẻ ảnh cũ nếu còn sót
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

def clean_chunk(raw_text: str, image_dict: dict = None):
    text = clean_markdown(raw_text, image_dict)
    chunks = chunks_for_rag(text)
    return chunks