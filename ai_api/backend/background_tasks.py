"""
Background Tasks: LLM nhỏ tóm tắt hội thoại → nạp ngược SQL.
"""
import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    AsyncSessionLocal,
    count_messages,
    fetch_all_messages_for_summary,
    fetch_summary,
    save_summary,
)

load_dotenv()

SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", "llama-3.1-8b-instant")
SUMMARIZE_TRIGGER_MESSAGES = int(os.getenv("SUMMARIZE_TRIGGER_MESSAGES", "20"))
SUMMARIZE_TRIGGER_CHARS = int(os.getenv("SUMMARIZE_TRIGGER_CHARS", "8000"))

_SUMMARY_PROMPT = PromptTemplate.from_template(
    """Bạn là hệ thống tóm tắt hội thoại. Tóm tắt ngắn gọn các điểm chính, giữ ngữ cảnh quan trọng cho câu hỏi tiếp theo.
Tóm tắt cũ (nếu có): {existing_summary}

Hội thoại:
{conversation}

Tóm tắt mới (tiếng Việt, tối đa 500 từ):"""
)


def _estimate_chars(messages) -> int:
    return sum(len(m.content) for m in messages)


async def should_summarize(db: AsyncSession, session_id: str) -> bool:
    total = await count_messages(db, session_id)
    if total < SUMMARIZE_TRIGGER_MESSAGES:
        return False

    summary = await fetch_summary(db, session_id)
    if summary and total - summary.message_count_at_summary < SUMMARIZE_TRIGGER_MESSAGES // 2:
        return False

    messages = await fetch_all_messages_for_summary(db, session_id)
    return _estimate_chars(messages) >= SUMMARIZE_TRIGGER_CHARS


async def summarize_conversation(session_id: str, user_id: str) -> None:
    """
    Chạy nền sau khi chat xong — dùng session DB riêng (không dùng request session).
    """
    async with AsyncSessionLocal() as db:
        try:
            if not await should_summarize(db, session_id):
                return

            messages = await fetch_all_messages_for_summary(db, session_id)
            existing = await fetch_summary(db, session_id)
            existing_text = existing.summary if existing else ""

            conversation = "\n".join(
                f"{m.role}: {m.content}" for m in messages[-40:]
            )

            llm = ChatGroq(
                model=SUMMARIZE_MODEL,
                temperature=0.3,
                api_key=os.getenv("API_KEY_GROQ"),
            )
            chain = _SUMMARY_PROMPT | llm
            result = await chain.ainvoke(
                {
                    "existing_summary": existing_text or "(chưa có)",
                    "conversation": conversation,
                }
            )
            summary_text = result.content.strip()
            total = await count_messages(db, session_id)
            await save_summary(db, session_id, user_id, summary_text, total)
        except Exception as exc:
            print(f"[summarize] session={session_id} error: {exc}")
