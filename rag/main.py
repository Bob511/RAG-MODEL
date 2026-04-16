from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# --- BƯỚC 1: ĐỊNH NGHĨA SCHEMA ---
class QuestionRequest(BaseModel):
    prompt: str

class AnswerResponse(BaseModel):
    answer: str
    status: str = "success"

# --- BƯỚC 2: VIẾT API NHẬN CÂU HỎI ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    # Tạm thời trả về câu trả lời giả (Mock data) 
    # để kiểm tra kết nối với Frontend trước
        user_query = request.prompt
                            
        return {
            "answer": f"Backend đã nhận được câu hỏi: {user_query}. Đang chờ AI xử lý...",
            "status": "success"
        }

# Endpoint cũ của Ngày 3
@app.get("/health")
async def health():
    return {"status": "ok"}