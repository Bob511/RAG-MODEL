from fastapi import FastAPI
from pathlib import Path
from llama_cloud import AsyncLlamaCloud
from dotenv import load_dotenv
from unit.chunk import adaptive_chunking_md
from unit.get_file import get_all_pdf_files
from unit.Validation import validate_and_cleanup_garbage
import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from unit.Validation import validate_and_cleanup_garbage
app = FastAPI()

# Mở CORS để UI chatbot có thể giao tiếp với server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

TEMP_DIR = "./temp_documents"
TEMP_DIR2 = "./Output_tempt_documents"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TEMP_DIR2, exist_ok=True)

# Giới hạn dung lượng (Ví dụ: 10MB)
MAX_FILE_SIZE = 40 * 1024 * 1024 

# ---------------------------------------------------------
# TIẾN TRÌNH XỬ LÝ NGẦM (Bước 3, 4, 5 sẽ nằm ở đây)
# ---------------------------------------------------------
async def background_document_pipeline(file_path: str, job_id: str, filename: str, output_filename: str):
    print(f"[{job_id}] Bắt đầu quy trình xử lý ngầm cho: {filename}")
    try:
        client = AsyncLlamaCloud()
        
        # SỬA LỖI 2: Phải mở file bằng open() trước khi đưa vào LlamaCloud
        with open(file_path, "rb") as f:
            cloud_file = await client.files.create(file=f, purpose="parse")
            
        result = await client.parsing.parse(
            file_id=cloud_file.id,
            tier="cost_effective",
            version="latest",
            output_options={
                "markdown": {
                    "annotate_links": False,
                    "inline_images": True,
                    "tables": { "merge_continued_tables": True, "output_tables_as_markdown": False }
                }
            }, 
            expand=["markdown_full"] 
        )
            
        Path(output_filename).write_text(result.markdown_full or "", encoding="utf-8")
        print("2")
        validate_and_cleanup_garbage(md_files=Path(output_filename))
        print("3")
        adaptive_chunking_md(Path(output_filename))
        print(f"[{job_id}] Đã chia chunk và lưu vào Vector Database thành công!")
        
        # Lưu ý: Trong thực tế, bạn sẽ cập nhật trạng thái job này vào một database (như SQLite)
        # để chatbot có thể truy vấn xem file đã sẵn sàng để hỏi đáp chưa.
        
    except Exception as e:
        print(f"[{job_id}] Lỗi hệ thống: {e}")


# ---------------------------------------------------------
# API ĐÓN NHẬN TỪ TRÌNH DUYỆT (Bước 2)
# ---------------------------------------------------------
@app.post("/api/upload")
async def receive_document(
    background_tasks: BackgroundTasks, 
    document: UploadFile = File(...)
):
    # 1. Kiểm tra định dạng cơ bản
    allowed_extensions = [".pdf", ".md", ".docx"]
    file_ext = os.path.splitext(document.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Định dạng file không được hỗ trợ.")
    # 2. Tạo mã định danh độc nhất cho file
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    temp_filepath = os.path.join(TEMP_DIR, f"{job_id}{file_ext}")
    output_temp_filepath = os.path.join(TEMP_DIR2, f"{job_id}.md")
    # 3. Đọc luồng byte và kiểm soát dung lượng (Spooling an toàn)
    file_size = 0
    try:
        with open(temp_filepath, "wb") as buffer:
            # Đọc từng khối nhỏ (chunk) 1MB để không làm nghẽn RAM
            while chunk := await document.read(1024 * 1024): 
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="File vượt quá 10MB.")
                buffer.write(chunk)
    except HTTPException:
        os.remove(temp_filepath)
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Lỗi khi ghi file xuống ổ đĩa.")
    finally:
        await document.close()

    # 4. Giao việc cho luồng ngầm
    background_tasks.add_task(background_document_pipeline, temp_filepath, job_id, document.filename, output_temp_filepath)
    
    # 5. Phản hồi lập tức cho UI
    return {
        "status": "success",
        "job_id": job_id,
        "message": "File đã được đưa vào hàng đợi xử lý."
    }
