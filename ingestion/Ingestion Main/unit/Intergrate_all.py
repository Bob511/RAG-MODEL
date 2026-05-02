import os
import gc
from .setup_converter import setup
from marker.convert import convert_single_pdf
from .Process_img import process_images_in_text
from .clean_chunk import clean_chunk
from .clean_chunk import clean_markdown

def intergrate_all(IMG_DIR: str, FINAL_MD: str, PDF_FILE: str):
    model_lst = setup()
    # 2. Convert & Save Images
    full_text, images_dict, out_meta = convert_single_pdf(PDF_FILE, model_lst)
    
    # Lưu ảnh vật lý để hàm regex có thể đọc được
    os.makedirs(IMG_DIR, exist_ok=True)
    for name, img in images_dict.items():
        img.save(os.path.join(IMG_DIR, name))

    # 3. LLM Processing (Thay thế ảnh bằng Text)
    full_text_with_ai = process_images_in_text(full_text, IMG_DIR)

    # 4. Clean & Chunk
    final_text = clean_markdown(full_text_with_ai, images_dict)
    chunks = clean_chunk(final_text)
    gc.collect()
    return chunks