import os
import uuid
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, Request, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from database import init_db, get_db, ChatHistory
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, wait_exponential, stop_after_attempt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)
from llm_client import BotAi

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HandshakeRequest(BaseModel):
    sender: str
    content: str

class QuestionRequest(BaseModel):
    session_id: str
    query: str
    user_id: str

class AnswerResponse(BaseModel):
    answer: str
    status: str = "success"

UPLOAD_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../infrastructure/volumes/users_uploads"))

ai_bot = BotAi()

@retry(
        wait=wait_exponential(multiplier=2, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
)

async def call_groq_ai_mock(query: str):
    print(f"Đang tiến hành gọi Groq API cho truy vấn: {query}")
    if asyncio.iscoroutinefunction(ai_bot.question):
        return await ai_bot.question(query)
    else:
        return await asyncio.to_thread(ai_bot.question, query)

@app.on_event("startup")
async def on_startup():
    await init_db()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/handshake")
async def handshake(request: HandshakeRequest):
    return {
        "status": "confirmed",
        "received_from": request.sender
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            return {"message": "Invalid file type. Only PDF accepted", "status": "error"}

        file_extension = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "original_name": file.filename, 
            "saved_name": unique_name, 
            "status": "success"
        }
    except Exception as e:
        return {"message": str(e), "status": "error"}

@app.post("/ask", response_model=AnswerResponse)
@limiter.limit("2/minute")
async def ask_question(http_request: Request, request: QuestionRequest, db: AsyncSession = Depends(get_db)):
    user_msg = ChatHistory(session_id=request.session_id, role="user", content=request.query)
    db.add(user_msg)
    await db.commit()

    ai_answer = await call_groq_ai_mock(request.query)
    
    ai_msg = ChatHistory(session_id=request.session_id, role="assistant", content=ai_answer)
    db.add(ai_msg)
    await db.commit()

    return {
        "answer": ai_answer,
        "status": "success"
    }