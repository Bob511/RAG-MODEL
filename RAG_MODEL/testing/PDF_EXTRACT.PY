import os
import re
import fitz
import ollama
import gc
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling_core.types.doc.document import TableItem, PictureItem, TextItem

# 1. CẤU HÌNH HỆ THỐNG
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1" # Tắt lỗi symlink Windows
FILE_INPUT = r"D:\RAG_MODEL\52500174(report_2)(2).pdf"
MODEL_VISION = "qwen2.5vl:3b"
BATCH_SIZE = 5  # Mỗi đợt xử lý 5 trang để tránh tràn RAM

# ==========================================
# CÁC HÀM BỔ TRỢ
# ==========================================

def get_content_from_qwen(image_path, is_matrix=True):
    """Dùng Qwen giải mã Ma trận/Phương trình/Bảng từ hình ảnh vùng."""
    prompt = (
        "Đây là công thức toán học (ma trận/hệ phương trình). Hãy chuyển nó thành mã LaTeX chuẩn, bọc trong $$. Chỉ trả về mã." 
        if is_matrix else 
        "Đây là một bảng dữ liệu. Hãy chuyển nó thành định dạng Markdown Table chuẩn."
    )
    try:
        response = ollama.generate(model=MODEL_VISION, prompt=prompt, images=[image_path])
        return response['response'].strip()
    except:
        return "\n[Lỗi giải mã Vision]\n"

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

def setup_docling():
    """Cấu hình Docling an toàn cho Windows."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(lang=["en", "vi"])
    pipeline_options.do_formula_enrichment = False  # Tắt để tránh lỗi Symlink/RAM
    pipeline_options.generate_picture_images = True # Quan trọng để lấy ảnh Ma trận
    
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

# ==========================================
# LUỒNG CHẠY CHÍNH
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(FILE_INPUT):
        print("❌ Không tìm thấy file PDF!")
        exit()

    # Bước 1: Khởi tạo
    converter = setup_docling()
    batches = split_pdf_into_batches(FILE_INPUT, pages_per_batch=BATCH_SIZE)
    all_final_chunks = []

    # Bước 2: Xử lý từng Batch
    for batch_path in batches:
        print(f"\n🚀 Đang xử lý batch: {batch_path}")
        result = converter.convert(batch_path)
        doc = result.document
        batch_markdown = ""

        # Bước 3: Duyệt từng Item trong Batch
        for item, _ in doc.iterate_items():
            # Xử lý Bảng
            if isinstance(item, TableItem):
                try:
                    item.get_image(doc).save("temp_v.png")
                    batch_markdown += f"\n{get_content_from_qwen('temp_v.png', False)}\n"
                except: pass
            
            # Xử lý Ma trận/Phương trình (Docling thường phân loại là Picture hoặc Formula)
            elif "formula" in item.self_ref.lower() or isinstance(item, PictureItem):
                try:
                    item.get_image(doc).save("temp_v.png")
                    batch_markdown += f"\n{get_content_from_qwen('temp_v.png', True)}\n"
                except: pass
            
            # Văn bản thường
            elif isinstance(item, TextItem):
                batch_markdown += f"{item.text}\n"

        # Bước 4: Làm sạch & Chunking ngay cho Batch đó
        clean_md = re.sub(r'\n{3,}', '\n\n', batch_markdown)
        chunks = re.split(r'\n(?=\#\# )', clean_md)
        all_final_chunks.extend([c.strip() for c in chunks if c.strip()])

        # Giải phóng bộ nhớ tạm
        if os.path.exists("temp_v.png"): os.remove("temp_v.png")
        gc.collect()

    # Bước 5: Kết quả
    print("\n" + "🌟" * 20)
    print(f"✅ HOÀN TẤT! Tổng cộng {len(all_final_chunks)} chunks.")
    print(f"📊 Các Ma trận/Phương trình đã được Qwen chuyển sang LaTeX chuẩn.")
    
    if all_final_chunks:
        print("\n--- DEMO CHUNK 1 ---")
        print(all_final_chunks[0][:1000])