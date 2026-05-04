import os
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, DateTime, Text

# 1. Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../infrastructure/volumes/sql_data/biz_rag.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# 2. Khởi tạo Engine & Session
engine = create_async_engine(DATABASE_URL, echo=False) # Tắt echo=True khi chạy thật để đỡ rối log
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# 3. Định nghĩa Models tối ưu
class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, nullable=True)
    
    # Quan hệ: Một user có nhiều document
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    original_name: Mapped[str] = mapped_column(String)
    storage_name: Mapped[str] = mapped_column(String, unique=True) # Đảm bảo tên lưu trữ không bao giờ trùng
    file_path: Mapped[str] = mapped_column(String) # Lưu đường dẫn tuyệt đối để AI1 tìm cho nhanh
    status: Mapped[str] = mapped_column(String, default="uploaded") # uploaded, processing, indexed
    upload_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    # Quan hệ ngược lại
    owner = relationship("User", back_populates="documents")

# 4. Dependency: Hàm lấy DB session cho FastAPI (Cực kỳ tối ưu)
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# 5. Khởi tạo DB
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)