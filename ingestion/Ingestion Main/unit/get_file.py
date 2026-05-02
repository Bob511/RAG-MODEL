import os
import glob

def get_all_pdf_files(folder_path):
    # Sử dụng glob để lọc ra các file có đuôi .pdf
    # recursive=True giúp tìm cả trong các thư mục con nếu cần
    query = os.path.join(folder_path, "*.pdf")
    return glob.glob(query)