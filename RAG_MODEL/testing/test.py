import os
import re
import gc
import fitz  # PyMuPDF
import ollama

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.pipeline_options import EasyOcrOptions
from docling_core.types.doc.document import PictureItem

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
FILE_PATH = r"D:\RAG_MODEL\52500174(report_2)(2).pdf"
IMAGE_DIR = "extracted_images"
MODEL_VISION = "qwen2.5-vl:3b"
BATCH_SIZE = 5 # Số trang cho mỗi lần xử lý (Chống tràn RAM)


def split_pdf_into_batches(input_pdf_path, output_folder="pdf_batches", pages_per_batch=5):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(input_pdf_path)
    total_pages = len(doc)
    batch_files = []
    
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
# 2. KHỞI TẠO DOCLING (CHUẨN V2.14+)
# ==========================================
def setup_docling_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True # Bật cắt ảnh
    pipeline_options.do_ocr = True      # Bật OCR
    do_formula_enrichment = True #chuyển các công thức toán học sang latex
    pipeline_options.ocr_options = EasyOcrOptions(lang=["en", "vi"]) # Hỗ trợ tiếng Việt
    
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

# ==========================================
# 3. GỌI VISION AI & CHÈN NỘI DUNG VÀO MD
# ==========================================
def process_vision_and_inject(md_text, image_path, image_name):
    try:
        response = ollama.generate(
            model=MODEL_VISION,
            prompt='Phân tích nội dung biểu đồ hoặc hình ảnh này một cách chi tiết. Nếu có số liệu, hãy liệt kê rõ ràng.',
            images=[image_path]
        )
        vision_text = response['response']
        print(f"   ✅ Đã phân tích xong {image_name}!")
        
        # Khối text sẽ chèn vào Markdown
        injection_block = f"\n\n> 📊 **[Phân tích từ Vision AI ({image_name})]:**\n> {vision_text}\n\n"
        
        # Thay thế placeholder của Docling bằng khối text trên
        pattern = rf"!\[.*?\]\(.*?{image_name}.*?\)|\[.*?{image_name}.*?\]"
        md_text = re.sub(pattern, injection_block, md_text)
        
        return md_text
    except Exception as e:
        print(f"   ❌ Lỗi AI: {e}")
        return md_text

# ==========================================
# 4. LÀM SẠCH VÀ CHUNKING
# ==========================================
def clean_markdown_with_regex(text):
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) # Xóa link rác
    text = re.sub(r'[^\w\s#*.|,\-\:\/=()]', '', text).strip() # Giữ lại ký tự hợp lệ
    return text

def chunks_from_pdf(text):
    # Chia theo Heading level 2 (##)
    raw_chunks = re.split(r'\n(?=\#\# )', text)
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]

# ==========================================
# LUỒNG ĐIỀU PHỐI CHÍNH (MAIN PROCESS)
# ==========================================
if __name__ == "__main__":
    os.makedirs(IMAGE_DIR, exist_ok=True)
    all_final_chunks = []
    
    try:
        # BƯỚC 1: Cắt PDF
        batch_files = split_pdf_into_batches(FILE_PATH, pages_per_batch=BATCH_SIZE)
        
        # BƯỚC 2: Khởi tạo "Bộ máy" Docling
        converter = setup_docling_converter()
        
        # BƯỚC 3: Xử lý từng file PDF nhỏ
        for batch_index, pdf_batch in enumerate(batch_files):
            print(f"\n🚀 ĐANG XỬ LÝ BATCH {batch_index + 1}/{len(batch_files)}: {pdf_batch}")
            
            # 3.1: Docling đọc file
            result = converter.convert(pdf_batch)
            doc = result.document
            
            # 3.2: Lưu ảnh vật lý từ Docling
            current_images = []
            pic_count = 0
            for element, _ in doc.iterate_items():
                if isinstance(element, PictureItem):
                    try:
                        img = element.get_image(doc)
                        if img:
                            # Tên ảnh duy nhất để không ghi đè giữa các batch
                            img_name = f"batch_{batch_index}_img_{pic_count}.png"
                            img_path = os.path.join(IMAGE_DIR, img_name)
                            img.save(img_path)
                            current_images.append((img_path, img_name))
                            pic_count += 1
                    except Exception as e:
                        print(f"   ⚠️ Lỗi lưu ảnh: {e}")

            # 3.3: Lấy Markdown gốc
            md_batch = doc.export_to_markdown()

            # 3.4: Cho AI đọc từng ảnh và chèn vào Markdown
            for path, name in current_images:
                md_batch = process_vision_and_inject(md_batch, path, name)

            # 3.5: Làm sạch và Chunking
            clean_md = clean_markdown_with_regex(md_batch)
            chunks = chunks_from_pdf(clean_md)
            
            # 3.6: Gom vào kho tổng
            all_final_chunks.extend(chunks)
            print(f"   ✔️ Hoàn tất Batch {batch_index + 1}. Thu được {len(chunks)} chunks.")
            
            # 3.7: Xả RAM quan trọng!
            del result, doc, md_batch, clean_md, chunks
            gc.collect()

        # BƯỚC 4: TỔNG KẾT
        print("\n" + "🌟" * 25)
        print(f"🎉 HOÀN THÀNH TOÀN BỘ QUY TRÌNH!")
        print(f"📊 Tổng số lượng chunks sẵn sàng: {len(all_final_chunks)}")
        print("🌟" * 25)
        a = 1
        for i in all_final_chunks:
            print(f"chunk ----{a}-----")
            print(i)
            print(40*"-")
            a+=1

    except Exception as e:
        print(f"\n❌ Lỗi hệ thống nghiêm trọng: {e}")