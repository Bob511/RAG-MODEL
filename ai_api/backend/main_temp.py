from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os

app = FastAPI()

class QuestionRequest(BaseModel):
    user_id: str
    prompt: str

class AnswerResponse(BaseModel):
    answer: str
    status: str = "success"

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            return {"message": "Invalid file type. Only PDF accepted", "status": "error"}
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"filename": file.filename, "status": "success"}
    except Exception as e:
        return {"message": str(e), "status": "error"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    return {
        "answer": f"Backend received query from {request.user_id}: {request.prompt}",
        "status": "success"
    }