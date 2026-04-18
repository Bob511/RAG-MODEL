import ollama
import os
import fitz
import gc
import re
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling_core.types.doc.document import TableItem, PictureItem, TextItem


file_path = r"D:\RAG_MODEL\52500174(report_2)(2).pdf"
MODEL_VISION = 'qwen2.5vl:3b'

# ==========================================
# 1. CẤU HÌNH & CHUẨN BỊ
# ==========================================
def setup_docling():
    pipeline_options = PdfPipelineOptions() #lệnh bật setting cho việc xử lý file PDF
    pipeline_options.do_ocr = True #bật nhận dạng ký tự quang học
    pipeline_options.do_formula_enrichment = False # Khi gặp các công thức toán ==> latex thay vì đọc thì bỏ qua để tối ưu thời gian
    pipeline_options.generate_picture_images = True # Khi scan đến những object đc OCR cho là images ==> extract 
    pipeline_options.ocr_options = EasyOcrOptions(lang=["en", "vi"]) # điều chỉnh ngôn ngữ scan
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

# phân chia file PDF để không bị tràn bộ nhớ 
import fitz
import os

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
        batch_toc = []
        for entry in full_toc:
            lvl, title, page = entry
            # Nếu mục lục nằm trong phạm vi trang của batch này
            if start_page < page <= end_page + 1:
                batch_toc.append([lvl, title, page - start_page])
        
        new_doc.set_toc(batch_toc)
        
        batch_filename = os.path.join(output_folder, f"batch_{i//pages_per_batch + 1}.pdf")
        new_doc.save(batch_filename)
        new_doc.close()
        batch_files.append(batch_filename)
        
    doc.close()
    return batch_files

# ==========================================
# 2. XỬ LÝ ĐA PHƯƠNG THỨC BẰNG VISION AI
# ==========================================
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
        return f"\n{prefix}{response['response'].strip()}\n\n"
    except Exception as e:
        return f"\n[Lỗi Vision AI: {e}]\n"

# ==========================================
# 3. LÀM SẠCH & CHUNKING
# ==========================================
def clean_markdown_with_regex(text):
    # CHỈ xóa các đường link rác (ví dụ: [image_1](...)). 
    # TUYỆT ĐỐI KHÔNG dùng [^\w\s...] để tránh xóa mất dấu \ của LaTeX
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) 
    return text.strip()

def chunks_from_pdf(text):
    raw_chunks = re.split(r'\n(?=\#\# )', text)
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]



# ==========================================
# LUỒNG THỰC THI CHÍNH
# ==========================================
if __name__ == "__main__":
    converter = setup_docling()
    list_small_pdfs = split_pdf_into_batches(file_path, pages_per_batch=3)
    all_final_chunks = []

    for small_pdf in list_small_pdfs:
        print(f"\n🚀 Đang quét Batch: {small_pdf}")
        result = converter.convert(small_pdf)
        doc = result.document
        batch_md = ""

        # DUYỆT QUA TỪNG PHẦN TỬ THAY VÌ LẤY TOÀN BỘ TEXT
        for item, _ in doc.iterate_items():
            
            # 1. Bắt Bảng biểu
            if isinstance(item, TableItem):
                print("   📊 Đang dịch Bảng...")
                try:
                    item.get_image(doc).save("temp_vision.png")
                    batch_md += ask_qwen_vision("temp_vision.png", "table")
                except: pass
                
            # 2. Bắt Ma trận / Phương trình
            elif "formula" in item.self_ref.lower():
                print("   🔢 Đang giải mã Ma trận/Toán học sang LaTeX...")
                try:
                    item.get_image(doc).save("temp_vision.png")
                    batch_md += ask_qwen_vision("temp_vision.png", "math")
                except: pass
                
            # 3. Bắt Biểu đồ / Hình ảnh (mà không phải là công thức)
            elif isinstance(item, PictureItem):
                print("   🖼️ Đang phân tích Biểu đồ/Ảnh...")
                try:
                    item.get_image(doc).save("temp_vision.png")
                    batch_md += ask_qwen_vision("temp_vision.png", "picture")
                except: pass
                
            # 4. Bắt Text thường
            elif isinstance(item, TextItem):
                batch_md += f"{item.text}\n"

        # Làm sạch và Cắt chunk cho batch này
        clean = clean_markdown_with_regex(batch_md)
        chunks = chunks_from_pdf(clean)
        all_final_chunks.extend(chunks)
        
        # Dọn dẹp
        if os.path.exists("temp_vision.png"): os.remove("temp_vision.png")
        gc.collect()

    print("\n" + "="*20 + " KẾT QUẢ CHUNKING " + "="*20)
    for a, chunk in enumerate(all_final_chunks, 1):
        print(f"\n--- CHUNK {a} ---")
        print(chunk)
        print("-" * 50)