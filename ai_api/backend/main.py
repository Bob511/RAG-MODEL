"""
FastAPI entry point — Tích hợp: Redis rate limit, semantic cache, SQL history, SSE streaming, background summarization.
"""
import asyncio
import os
import shutil
import sys
import uuid
import json

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from background_tasks import summarize_conversation
from database import (
    ChatHistory,
    build_context_prompt,
    fetch_history,
    fetch_summary,
    get_db,
    init_db,
)
from redis_client import check_rate_limit, close_redis, redis_health_check
from semantic_cache import invalidate_session_cache, lookup_cache, store_cache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)
from llm_client import BotAi  # noqa: E402

app = FastAPI(title="RAG API (Production)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

UPLOAD_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../../infrastructure/volumes/users_uploads")
)

ai_bot = BotAi()


class HandshakeRequest(BaseModel):
    sender: str
    content: str


class QuestionRequest(BaseModel):
    session_id: str
    query: str
    user_id: str


async def enforce_rate_limit(user_id: str) -> None:
    allowed, _ = await check_rate_limit(
        identifier=user_id or "anonymous",
        limit=RATE_LIMIT_PER_MINUTE,
        window_seconds=RATE_LIMIT_WINDOW,
        prefix="ratelimit:user",
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Vượt giới hạn {RATE_LIMIT_PER_MINUTE} request/{RATE_LIMIT_WINDOW}s",
        )


async def save_chat_and_cache_and_summarize(
    session_id: str, user_id: str, query: str, answer: str, is_cached: bool
) -> None:
    """Hậu xử lý chạy ngầm: Lưu ChatHistory, lưu cache và tóm tắt hội thoại."""
    from database import AsyncSessionLocal, ChatHistory
    from semantic_cache import store_cache
    from background_tasks import summarize_conversation

    # 1. Lưu chuỗi answer hoàn chỉnh vào ChatHistory
    async with AsyncSessionLocal() as db:
        try:
            user_msg = ChatHistory(
                session_id=session_id,
                user_id=user_id,
                role="user",
                content=query,
            )
            ai_msg = ChatHistory(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=answer,
            )
            db.add_all([user_msg, ai_msg])
            await db.commit()
        except Exception as e:
            print(f"[post-stream] Lỗi lưu ChatHistory: {e}")

    # 2. Lưu cặp query/answer vào store_cache (nếu chưa có trong cache)
    if not is_cached:
        try:
            await store_cache(session_id, query, answer)
        except Exception as e:
            print(f"[post-stream] Lỗi lưu semantic cache: {e}")

    # 3. Kích hoạt summarize_conversation
    try:
        await summarize_conversation(session_id, user_id)
    except Exception as e:
        print(f"[post-stream] Lỗi tóm tắt hội thoại: {e}")


@app.on_event("startup")
async def on_startup():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    await init_db()


@app.on_event("shutdown")
async def on_shutdown():
    await close_redis()


@app.get("/health")
async def health_check():
    redis_ok = await redis_health_check()
    return {"status": "ok" if redis_ok else "degraded", "redis": redis_ok}


@app.post("/handshake")
async def handshake(request: HandshakeRequest):
    return {"status": "confirmed", "received_from": request.sender}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str | None = Query(default=None),
):
    try:
        if not file.filename.endswith(".pdf"):
            return {"message": "Invalid file type. Only PDF accepted", "status": "error"}

        file_extension = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if session_id:
            await invalidate_session_cache(session_id)

        return {
            "original_name": file.filename,
            "saved_name": unique_name,
            "status": "success",
        }
    except Exception as e:
        return {"message": str(e), "status": "error"}


@app.post("/ask")
async def ask_question(
    request: QuestionRequest,
    db: AsyncSession = Depends(get_db),
):
    # 1. Kiểm tra rate limit
    await enforce_rate_limit(request.user_id)

    # 2. Kiểm tra semantic cache
    cached_answer = await lookup_cache(request.session_id, request.query)
    
    if cached_answer:
        # Cache hit: Giả lập stream yield các từ trong cache ra
        async def cached_streamer():
            accumulated = []
            try:
                words = cached_answer.split(" ")
                for i, word in enumerate(words):
                    token = word + (" " if i < len(words) - 1 else "")
                    accumulated.append(token)
                    yield f"data: {token}\n\n"
                    await asyncio.sleep(0.02)
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
            finally:
                full_answer = "".join(accumulated)
                if full_answer:
                    asyncio.create_task(
                        save_chat_and_cache_and_summarize(
                            session_id=request.session_id,
                            user_id=request.user_id,
                            query=request.query,
                            answer=full_answer,
                            is_cached=True
                        )
                    )
        return StreamingResponse(cached_streamer(), media_type="text/event-stream")

    # 3. Cache miss: Gọi LLM qua Langchain astream()
    history = await fetch_history(db, request.session_id, limit=10, exclude_latest=0)
    summary = await fetch_summary(db, request.session_id)
    enriched_query = build_context_prompt(history, summary, request.query)

    async def llm_streamer():
        accumulated = []
        try:
            # Thực hiện hybrid search giống BotAi.question nhưng hỗ trợ stream
            if not ai_bot.hybrid_retrievers:
                ai_bot.hybrid_search()
            
            sub_questions = await ai_bot.decompose.ainvoke({"cau_hoi": enriched_query})
            sub_ques_query = [q.replace("-", "").strip() for q in sub_questions.content.split('\n') if q.strip()]
            
            relevant_task = [ai_bot.hybrid_retrievers.ainvoke(sq) for sq in sub_ques_query]
            gather_docs = await asyncio.gather(*relevant_task)
            
            store_relevant_docs = []
            for docs in gather_docs:
                store_relevant_docs.extend(docs)
                
            unique_docs = []
            seen = set()
            for doc in store_relevant_docs:
                text = doc.page_content
                if text not in seen:
                    seen.add(text)
                    unique_docs.append(doc)
                    
            final_docs = ai_bot.hybrid_retrievers.base_compressor.compress_documents(
                documents=unique_docs, query=enriched_query
            )
            context_part = []
            for doc in final_docs:
                source = doc.metadata.get("source", "unkown")
                index = doc.metadata.get("index", "unknown")
                content = doc.page_content
                context_part.append(f"source: {source} | index: {index} \n {content}")
            context = "\n".join(context_part)
            
            # Streaming từ deploy chain
            async for chunk in ai_bot.deploy.astream({"ngu_canh": context, "cau_hoi": enriched_query}):
                token = chunk.content
                accumulated.append(token)
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
        finally:
            full_answer = "".join(accumulated)
            if full_answer:
                asyncio.create_task(
                    save_chat_and_cache_and_summarize(
                        session_id=request.session_id,
                        user_id=request.user_id,
                        query=request.query,
                        answer=full_answer,
                        is_cached=False
                    )
                )

    return StreamingResponse(llm_streamer(), media_type="text/event-stream")
