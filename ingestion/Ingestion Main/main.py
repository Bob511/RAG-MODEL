import os
import shutil
from unit.get_file import get_all_pdf_files
from unit.main2 import intergrate_all
from unit.database_save import save_chunks_to_chromadb_cloud



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
        IMG_DIR = "./extracted_images"
        FINAL_MD = "./final_output.md"
        chunks = intergrate_all(PDF_FILE=PDF_FILE, IMG_DIR=IMG_DIR, FINAL_MD=FINAL_MD)
        save_chunks_to_chromadb_cloud(chunks)
    except Exception as e:
        print(f"error: {e}")
    finally:
        a = input("Are you want to keep img_dir(Y/N): ")
        if os.path.exists(IMG_DIR) and a == "N":
            shutil.rmtree(IMG_DIR)
        b = input("Are you want to keep file output md(Y/N): ")
        if os.path.exists(FINAL_MD) and b == "N":
            os.remove(FINAL_MD)
        
