# Pipeline:
- Get file function --> choose file and check exists available --> setting download Model needed to convert --> extract image and analyse --> insert back to the markdown

# Detail:
## Get file fucntion.
- Basicially it's allow you choose file on your desktop optionally and in terminal by typting the folder name and name of file at the folder in disc 

- Reason for this  func: lazy to path and find the name file variable

- Function:

def get_all_pdf_files(folder_path):
    query = os.path.join(folder_path, "*.pdf")
    return glob.glob(query)
-  tham số nhận vào là tên folder ở ổ đĩa hiện tại, hàm os.path.join(folder, ...) có nghĩa là ghép 2 tên lại với nhau để thành một path hoàn chỉnh, 
- Dấu sao ở đây mang nghĩa là một chuỗi ký tự bất kỳ, *.pdf có nghĩa là bất cứ file nào có đuôi bằng .pdf đều hợp lệ
- glob.glob() mang tác dụng như search


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
## 
