import os
import asyncio
from pathlib import Path
from llama_cloud import AsyncLlamaCloud
from dotenv import load_dotenv
from unit.chunk import adaptive_chunking_md
from unit.get_file import get_all_pdf_files
from unit.Validation import validate_and_cleanup_garbage
load_dotenv()  # Tải biến môi trường từ file .env
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
async def main(FILE_PATH, OUTPUT_FILE):
    client = AsyncLlamaCloud()
    file = await client.files.create(file=FILE_PATH, purpose="parse")
    result = await client.parsing.parse(
        file_id=file.id,
        tier="cost_effective",
        version="latest",
        output_options= {
            "markdown": {
            "annotate_links": False,
            "inline_images": True,
            "tables": { "merge_continued_tables": True, "output_tables_as_markdown": False }}}, 
        expand=["markdown_full"] )
    Path(OUTPUT_FILE).write_text(result.markdown_full or "", encoding="utf-8")
    validate_and_cleanup_garbage(OUTPUT_FILE)
    adaptive_chunking_md(OUTPUT_FILE)
if __name__ == "__main__":
    try:
        input_folder = "temp_documents"
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
            OUTPUT_FILE = Path(r"D:\ragmodel\data_rag_output\report_cleaned.md")
            asyncio.run(main(PDF_FILE, OUTPUT_FILE))
    except Exception as e:
        print(f"error: {e}")
    
    
