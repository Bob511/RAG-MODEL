#Các thư viện sử dụng
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from database import init_db
import uvicorn
import uuid
import shutil
import os

app = FastAPI() #Khởi tạo app để bắt đầu nhận các yêu cầu từ Web

class QuestionRequest(BaseModel): #Định nghĩa khuôn mẫu câu hỏi: Web2 gửi lên phải đúng 2 dữ liệu này
    user_id: str #Người dùng
    prompt: str #Nội dung

class AnswerResponse(BaseModel): #Schema trả lời yêu cầu 2 dữ liệu
    answer: str #Nội dung trả lời
    status: str = "success" #Trạng thái thành công hoặc khác

#upload trong thư mục infrastructure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../infrastructure/volumes/users_uploads"))

#endpoint health
@app.get("/health")
async def health_check():
    return {"status": "ok"}

#endpoint upload
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try: #Thử nếu không được thì lỗi
        if not file.filename.endswith(".pdf"): #Phải là pdf
            return {"message": "Invalid file type. Only PDF accepted", "status": "error"}

        #Lưu tên file nhưng tên khác với tên người dùng đã gửi
        file_extension = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)
        
        #Lưu vào upload
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "original_name": file.filename, 
            "saved_name": unique_name, 
            "status": "success"}
    except Exception as e: #Lỗi
        return {"message": str(e), "status": "error"}

#endpoint ask
@app.post("/ask", response_model=AnswerResponse) #Đảm bảo câu trả lời gửi về cho Web2 luôn đúng định dạng
async def ask_question(request: QuestionRequest):
    #MOCK - Thay đổi thành câu trả lời thật từ AI
    return {
        "answer": f"Received query from {request.user_id}: {request.prompt}",
        "status": "success"
    }

#
@app.on_event("startup")
async def on_startup():
    # Khi App bắt đầu chạy, lệnh này sẽ tạo file biz_rag.db và các bảng nếu chưa có
    await init_db()
    print("--- Database đã được khởi tạo thành công! ---")