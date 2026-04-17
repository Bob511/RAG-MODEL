from fastapi import FastAPI, UploadFile, File  # Thêm UploadFile và File ở đây
from pydantic import BaseModel
import shutil # Thêm thư viện để copy file
import os     # Thêm thư viện để tạo thư mục

app = FastAPI()

# --- BƯỚC 1: ĐỊNH NGHĨA SCHEMA ---
class QuestionRequest(BaseModel):
    prompt: str

class AnswerResponse(BaseModel):
    answer: str
    status: str = "success"

# --- MỚI: CẤU HÌNH THƯ MỤC LƯU TRỮ ---
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR) # Tự động tạo thư mục 'uploads' nếu chưa có

# --- BƯỚC 2: VIẾT API NHẬN FILE PDF (MỚI) ---
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Kiểm tra nếu không phải file PDF thì từ chối
        if not file.filename.endswith(".pdf"):
            return {"message": "Chỉ chấp nhận file định dạng .pdf", "status": "error"}

        # Tạo đường dẫn: uploads/ten_file.pdf
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Lưu file vào thư mục uploads
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "filename": file.filename,
            "message": "Backend đã nhận và lưu file thành công!",
            "status": "success"
        }
    except Exception as e:
        return {"message": f"Lỗi khi lưu file: {str(e)}", "status": "error"}

# --- BƯỚC 2.1: VIẾT API NHẬN CÂU HỎI (GIỮ NGUYÊN) ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    user_query = request.prompt
    return {
        "answer": f"Backend đã nhận được câu hỏi: {user_query}. Đang chờ AI xử lý...",
        "status": "success"
    }

# Endpoint cũ của Ngày 3
@app.get("/health")
async def health():
    return {"status": "ok"}