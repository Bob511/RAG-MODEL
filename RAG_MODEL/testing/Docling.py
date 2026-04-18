import ollama
import os
import re
import fitz  # PyMuPDF
import gc
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# CẤU HÌNH
# ==========================================
FILE_PATH = r"D:\RAG_MODEL\52500174(report_2)(2).pdf"
MODEL_VISION = "qwen2.5vl:3b"

# ==========================================
# HÀM CỨU HỘ: DÙNG VISION ĐỂ GIẢI MÃ TOÁN/BẢNG
# ==========================================
def rescue_page_with_vision(pdf_batch_path, page_index_in_batch):
    """
    Chụp ảnh trang cụ thể trong batch và nhờ Qwen dịch sang Markdown + LaTeX
    """
    try:
        doc = fitz.open(pdf_batch_path)
        page = doc.load_page(page_index_in_batch)
        pix = page.get_pixmap(dpi=300) # DPI cao để AI nhìn rõ công thức
        img_path = f"temp_rescue_p{page_index_in_batch}.png"
        pix.save(img_path)
        doc.close()

        print(f"   🕵️ Đang nhờ {MODEL_VISION} giải mã trang có lỗi công thức/bảng...")
        
        prompt = """Chuyển đổi nội dung bức ảnh này thành văn bản Markdown.
        1. Các công thức toán học PHẢI dùng LaTeX (kẹp trong $$ hoặc $).
        2. Nếu có bảng biểu, hãy trình bày dạng Markdown Table chuẩn.
        3. Giữ nguyên cấu trúc phân cấp tiêu đề (##).
        Chỉ trả về Markdown, không giải thích thêm."""

        response = ollama.generate(
            model=MODEL_VISION,
            prompt=prompt,
            images=[img_path]
        )
        
        # Xóa ảnh tạm sau khi dùng
        if os.path.exists(img_path):
            os.remove(img_path)
            
        return response['response']
    except Exception as e:
        print(f"   ❌ Lỗi cứu hộ Vision: {e}")
        return ""

# ==========================================
# CÁC HÀM XỬ LÝ VĂN BẢN
# ==========================================
def setup_docling():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(lang=["en", "vi"])
    # Thử bật nhận diện công thức, nếu lỗi nó sẽ ra "--formula-not-decoded--"
    pipeline_options.do_formula_enrichment = False
    
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

def clean_markdown_with_regex(text):
    # Xóa link ảnh/file rác
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) 
    # Lưu ý: Tôi bỏ dòng xóa ký tự đặc biệt để bảo vệ mã LaTeX ($$)
    return text.strip()

def chunks_from_pdf(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "])
    print(text_splitter)
    # split_documents giúp giữ lại metadata (như tên file, số trang)
    splits = text_splitter.split_documents([text])
    return splits

def split_pdf_into_batches(input_pdf_path, output_folder="pdf_batches", pages_per_batch=3):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(input_pdf_path)
    full_toc = doc.get_toc() # Lấy mục lục của file gốc
    batch_files = []

    for i in range(0, len(doc), pages_per_batch):
        start_page = i
        end_page = min(i + pages_per_batch - 1, len(doc) - 1)
        
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
        
        # Lọc và điều chỉnh lại mục lục cho file batch này
        # Lọc và điều chỉnh lại mục lục cho file batch này
        batch_toc = []
        for entry in full_toc:
            lvl, title, page = entry
            if start_page < page <= end_page + 1:
                batch_toc.append([lvl, title, page - start_page])
        
        # --- FIX LỖI VALUEERROR TẠI ĐÂY ---
        if batch_toc:
            valid_toc = []
            prev_level = 0
            for item in batch_toc:
                lvl, title, page = item
                
                # 1. Phần tử đầu tiên bắt buộc phải là level 1
                if not valid_toc:
                    lvl = 1
                # 2. Các phần tử sau không được nhảy cóc (vd: prev đang là 1 thì tiếp theo tối đa là 2)
                else:
                    lvl = min(lvl, prev_level + 1)
                
                valid_toc.append([lvl, title, page])
                prev_level = lvl
            
            new_doc.set_toc(valid_toc)
        else:
            # Nếu batch này không có mục lục nào thì bỏ qua
            new_doc.set_toc([])
        # -----------------------------------
        
        batch_filename = os.path.join(output_folder, f"batch_{i//pages_per_batch + 1}.pdf")
        new_doc.save(batch_filename)
        new_doc.close()
    return batch_files

def embbeded(splits):
    vector_results = []
    for i, chunk in enumerate(splits):
        response = ollama.embed(
            model='qwen3-embedding:0.6b',
            input=chunk.page_content
        )
        vector_results.append({
            "id": i,
            "text": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": response['embeddings'][0]
        })
    
    return vector_results

# ==========================================
# LUỒNG CHẠY CHÍNH
# ==========================================
if __name__ == "__main__":
    converter = setup_docling()
    list_small_pdfs = split_pdf_into_batches(FILE_PATH, pages_per_batch=5)
    all_final_chunks = []

    for small_pdf in list_small_pdfs:
        print(f"\n🚀 Đang quét Batch: {small_pdf}")
        
        result = converter.convert(small_pdf)
        
        # Kiểm tra từng trang trong batch xem có lỗi giải mã không
        # Nếu Docling trả về "--formula-not-decoded--", ta chụp trang đó cho Qwen đọc
        md_output = result.document.export_to_markdown()
        
        if "--formula-not-decoded--" in md_output or "table" in md_output.lower():
            print("   ⚠️ Phát hiện công thức lỗi hoặc bảng phức tạp. Bắt đầu cứu hộ từng trang...")
            
            # Mở lại batch để đếm trang
            batch_doc = fitz.open(small_pdf)
            rescued_parts = []
            
            for p_idx in range(len(batch_doc)):
                page_content = rescue_page_with_vision(small_pdf, p_idx)
                rescued_parts.append(page_content)
            
            batch_doc.close()
            final_md_of_batch = "\n\n".join(rescued_parts)
        else:
            final_md_of_batch = md_output

        # Làm sạch và Chunking
        clean = clean_markdown_with_regex(final_md_of_batch)
        chunks = chunks_from_pdf(clean)
        all_final_chunks.extend(chunks)
        
        gc.collect()

    # IN KẾT QUẢ
    print("\n" + "="*20 + " KẾT QUẢ CHUNKING " + "="*20)
    for a, chunk in enumerate(all_final_chunks, 1):
        print(f"\n--- CHUNK {a} ---")
        print(chunk)
        print("-" * 50)
        
    vector = embbeded(all_final_chunks)
    for i in vector:
        print(f"vector {i} = {len(i)}")