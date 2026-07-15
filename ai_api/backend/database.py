"""
SQL layer với connection pooling + lịch sử/tóm tắt hội thoại theo chuẩn 3NF.
"""
import os
import uuid
from datetime import datetime
from typing import Optional, List

from dotenv import load_dotenv
from sqlalchemy import String, ForeignKey, DateTime, Text, Integer, select, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
POOL_SIZE = int(os.getenv("SQL_POOL_SIZE", "3"))
MAX_OVERFLOW = int(os.getenv("SQL_MAX_OVERFLOW", "5"))
POOL_RECYCLE = int(os.getenv("SQL_POOL_RECYCLE", "1800"))

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=POOL_RECYCLE,
)
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    
    documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="owner", cascade="all, delete-orphan"
    )


class Author(Base):
    __tablename__ = "authors"
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    
    documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="author"
    )


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    author_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("authors.id"), nullable=True)
    original_name: Mapped[str] = mapped_column(String)
    storage_name: Mapped[str] = mapped_column(String, unique=True)
    file_path: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="uploaded")
    upload_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    
    owner: Mapped["User"] = relationship("User", back_populates="documents")
    author: Mapped[Optional["Author"]] = relationship("Author", back_populates="documents")
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(String, ForeignKey("documents.id"))
    content: Mapped[str] = mapped_column(Text)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(String, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ConversationSummary(Base):
    """Tóm tắt rolling theo session — upsert một dòng/session."""
    __tablename__ = "conversation_summaries"
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    message_count_at_summary: Mapped[int] = mapped_column(default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def fetch_history(
    db: AsyncSession,
    session_id: str,
    limit: int = 10,
    exclude_latest: int = 0,
) -> list[ChatHistory]:
    """Kéo N message gần nhất của session."""
    stmt = (
        select(ChatHistory)
        .where(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(limit + exclude_latest)
    )
    result = await db.execute(stmt)
    rows = list(result.scalars().all())
    if exclude_latest and rows:
        rows = rows[exclude_latest:]
    rows.reverse()
    return rows


async def fetch_summary(
    db: AsyncSession, session_id: str
) -> Optional[ConversationSummary]:
    stmt = select(ConversationSummary).where(
        ConversationSummary.session_id == session_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def count_messages(db: AsyncSession, session_id: str) -> int:
    stmt = select(func.count()).select_from(ChatHistory).where(
        ChatHistory.session_id == session_id
    )
    result = await db.execute(stmt)
    return result.scalar_one()


async def save_summary(
    db: AsyncSession,
    session_id: str,
    user_id: str,
    summary_text: str,
    message_count: int,
) -> ConversationSummary:
    existing = await fetch_summary(db, session_id)
    if existing:
        existing.summary = summary_text
        existing.message_count_at_summary = message_count
        existing.user_id = user_id
        existing.updated_at = datetime.now()
        await db.commit()
        await db.refresh(existing)
        return existing

    row = ConversationSummary(
        session_id=session_id,
        user_id=user_id,
        summary=summary_text,
        message_count_at_summary=message_count,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def fetch_all_messages_for_summary(
    db: AsyncSession, session_id: str
) -> list[ChatHistory]:
    stmt = (
        select(ChatHistory)
        .where(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.asc())
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


def build_context_prompt(
    history: list[ChatHistory],
    summary: Optional[ConversationSummary],
    current_query: str,
) -> str:
    """Ghép summary + history vào prompt multi-turn."""
    parts: list[str] = []
    if summary and summary.summary.strip():
        parts.append(f"[Tóm tắt hội thoại trước]\n{summary.summary}")
    if history:
        lines = [f"{m.role}: {m.content}" for m in history]
        parts.append("[Lịch sử gần đây]\n" + "\n".join(lines))
    parts.append(f"user: {current_query}")
    return "\n\n".join(parts)
