"""
LLM CLIENT - Lõi AI cho hệ thống RAG (Phiên bản 3.0)
=======================================================
MỤC TIÊU: Nhận câu hỏi user, tìm kiếm thông tin liên quan trong DB, trả lời bằng LLM.

DATAFLOW CHI TIẾT:
  1. FastAPI nhận request (query + session_id + user_id + db: AsyncSession)
  2. BotAi.question() được gọi
  3. AgentRouter (Gemini) phân tích: intent + source_filter + 3 sub_queries (1 request)
  4. Hybrid Search song song:
       a. ChromaDB Cloud (Cosine Similarity) → dùng where_filter nếu có source
       b. PostgreSQL FTS (tsvector @@ tsquery) → trả về danh sách chunk_id
  5. Gộp 2 danh sách chunk_id bằng RRF (Reciprocal Rank Fusion)
  6. Truy vấn PostgreSQL lấy Text thô theo chunk_id đã gộp (JOIN documents)
  7. Jina Rerank với số chunk động (3 hoặc 20 tuỳ intent) + Score Threshold < 0.5
  8. Lấy 10 tin nhắn gần nhất từ PostgreSQL (SQLAlchemy ORM) → context window
  9. Gộp context + chat history + câu hỏi → nạp vào LLM Groq (streaming)
 10. Lưu câu hỏi + câu trả lời vào chat_history (SQLAlchemy ORM)

THAY ĐỔI SO VỚI PHIÊN BẢN 2.0:
  - BỎ:  _decompose_question() dùng Groq (tốn quota 30 RPM)
  - BỎ:  asyncpg raw SQL cho chat_history
  - THÊM: AgentRouter (Gemini) = Router + Decompose trong 1 request
  - THÊM: source_filter → where filter cho ChromaDB theo tên file/URL
  - THÊM: Score Threshold 0.5 sau Jina Rerank (loại chunk kém liên quan)
  - SỬA:  _fetch_chat_history() → dùng SQLAlchemy AsyncSession (ORM)
  - SỬA:  _save_chat_history() → dùng SQLAlchemy AsyncSession (ORM)
  - SỬA:  question() nhận thêm db: AsyncSession từ FastAPI Dependency
"""

import os, time, asyncio, chromadb, asyncpg
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_compressors import JinaRerank
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from llm_router import AgentRouter
load_dotenv()


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CHROMA_COLLECTION   = "VectorChunk"
CHROMA_DB_NAME      = "VectorManagement"
CHAT_HISTORY_LIMIT  = 10    # Số tin nhắn lịch sử đưa vào context window
JINA_SCORE_THRESHOLD = 0.5  # Loại bỏ chunk có relevance_score < ngưỡng này


# ─────────────────────────────────────────────
# HÀM TIỆN ÍCH: Reciprocal Rank Fusion (RRF)
# ─────────────────────────────────────────────
def reciprocal_rank_fusion(
    chroma_ids: list[str],
    sql_ids: list[str],
    k: int = 60
) -> list[str]:
    """
    Gộp 2 danh sách ID từ Chroma và SQL bằng thuật toán RRF.

    Công thức: score(d) = Σ 1 / (k + rank(d))
    - k=60 là hằng số chuẩn của RRF (theo bài báo Cormack 2009)
    - Chunk xuất hiện trong cả 2 nguồn → điểm cộng dồn → xếp hạng cao hơn

    Args:
        chroma_ids: Danh sách chunk_id từ ChromaDB (đã xếp theo cosine similarity)
        sql_ids:    Danh sách chunk_id từ PostgreSQL FTS (đã xếp theo FTS score)
        k:          Hằng số RRF (default=60)

    Returns:
        Danh sách chunk_id đã gộp và sắp xếp theo điểm RRF giảm dần
    """
    scores: dict[str, float] = {}

    for rank, chunk_id in enumerate(chroma_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, chunk_id in enumerate(sql_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in sorted_ids]


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────
class BotAi:
    """
    BotAi: Lõi xử lý AI cho toàn bộ pipeline RAG (v3.0).

    Attributes:
        llm:            ChatGroq - model LLM chính để trả lời (bảo toàn 30 RPM)
        router:         AgentRouter - phân tích intent + decompose (Gemini)
        deploy:         Chain trả lời chính (PromptTemplate | llm)
        chroma_client:  Kết nối ChromaDB Cloud
        chroma_col:     Collection VectorChunk trong ChromaDB
        jina:           JinaRerank - lọc chunk liên quan nhất (Score Threshold)
        _pg_pool:       asyncpg Connection Pool cho PostgreSQL FTS
    """

    def __init__(self):
        # ── 1. Khởi tạo LLM Groq (chỉ dùng cho bước trả lời cuối cùng) ──
        self.llm = ChatGroq(
            model=os.getenv("MODEL"),
            temperature=0.7,
            api_key=os.getenv("API_KEY_GROQ")
        )

        # ── 2. Prompt trả lời chính ──
        # {lich_su}: 10 tin nhắn gần nhất từ chat_history (SQLAlchemy ORM)
        # {ngu_canh}: top chunks sau Jina Rerank + Score Threshold
        # {cau_hoi}: câu hỏi gốc của user
        prompt_answer = (
            "Bạn là trợ lý AI chuyên phân tích tài liệu. "
            "Hãy trả lời bằng ngôn ngữ của câu hỏi và trích dẫn nguồn cụ thể.\n\n"
            "Lịch sử hội thoại gần đây:\n{lich_su}\n\n"
            "Ngữ cảnh từ tài liệu:\n{ngu_canh}\n\n"
            "Câu hỏi: {cau_hoi}\n"
            "Trả lời:"
        )
        self.template = PromptTemplate(
            input_variables=["lich_su", "ngu_canh", "cau_hoi"],
            template=prompt_answer
        )
        self.deploy = self.template | self.llm

        # ── 3. Agent Router (Gemini) - thay thế _decompose_question Groq ──
        # Gộp: Intent Detection + Metadata Extraction + Query Decomposition
        # Tiết kiệm 1 Groq request mỗi lần user hỏi
        self.router = AgentRouter()

        # ── 4. Kết nối ChromaDB Cloud ──
        self.chroma_client = chromadb.CloudClient(
            tenant=os.getenv("TENANT"),
            api_key=os.getenv("CHROMA_API"),
            database=CHROMA_DB_NAME
        )
        self.chroma_col = self.chroma_client.get_collection(name=CHROMA_COLLECTION)

        # ── 5. Jina Rerank ──
        # top_n sẽ được điều chỉnh động dựa theo intent từ AgentRouter
        self.jina = JinaRerank(
            jina_api_key=os.getenv("JINA_API"),
            top_n=3  # Sẽ bị override mỗi lần gọi
        )

        # ── 6. asyncpg pool cho PostgreSQL FTS (vẫn dùng asyncpg cho FTS search) ──
        # Ghi chú: Chat history sẽ dùng SQLAlchemy ORM (từ FastAPI dependency)
        # asyncpg chỉ còn dùng cho _sql_fts_search và _fetch_chunks_by_ids
        self._pg_pool: asyncpg.Pool | None = None

    # ─────────────────────────────────────────
    # KHỞI TẠO & ĐÓNG ASYNCPG POOL
    # ─────────────────────────────────────────
    async def _get_pg_pool(self) -> asyncpg.Pool:
        """
        Trả về asyncpg pool, khởi tạo lazily nếu chưa có.
        asyncpg dùng connection string riêng (không có prefix +asyncpg của SQLAlchemy).
        DATABASE_URL cần format: postgresql://user:pass@host:port/dbname
        """
        if self._pg_pool is None:
            raw_url = os.getenv("DATABASE_URL", "").replace(
                "postgresql+asyncpg://", "postgresql://"
            )
            self._pg_pool = await asyncpg.create_pool(
                dsn=raw_url,
                min_size=2,
                max_size=10,
                command_timeout=30
            )
        return self._pg_pool

    async def close(self):
        """Đóng pool khi shutdown (gọi trong @app.on_event('shutdown'))."""
        if self._pg_pool:
            await self._pg_pool.close()

    # ─────────────────────────────────────────
    # STEP A: SEMANTIC SEARCH (ChromaDB Cloud)
    # ─────────────────────────────────────────
    async def _chroma_search(
        self,
        sub_queries: list[str],
        top_k: int,
        source_filter: Optional[str] = None
    ) -> list[str]:
        """
        Tìm kiếm vector trên ChromaDB Cloud với Metadata Pre-filtering.

        Dataflow:
          sub_queries (list[str]) + source_filter (str|None)
            → chromadb.Collection.query(query_texts=[...], where={...}, n_results=top_k)
            → ChromaDB tự embed bằng model mặc định
            → Lọc metadata TRƯỚC khi đo Cosine (nếu source_filter != None)
            → Trả về chunk_ids đã sắp xếp theo độ tương đồng giảm dần

        Args:
            sub_queries:   3 câu hỏi phụ từ AgentRouter
            top_k:         Số kết quả tối đa (động theo intent: 10 hoặc 40)
            source_filter: Tên file hoặc URL để lọc metadata (None = không lọc)

        Note:
            - Metadata Pre-filtering xảy ra TRƯỚC khi tính Cosine → tiết kiệm tài nguyên
            - Metadata trong ChromaDB phải có trường "source" khớp với tên file
        """
        loop = asyncio.get_running_loop()

        # Xây dựng where_filter cho ChromaDB nếu có source
        where_filter = None
        if source_filter:
            where_filter = {"source": {"$eq": source_filter}}
            print(f"   🔽 ChromaDB Metadata Filter: source='{source_filter}'")

        def _sync_query():
            kwargs = {
                "query_texts": sub_queries,
                "n_results":   top_k,
                "include":     [],  # Chỉ lấy IDs, không lấy documents hay embeddings
            }
            if where_filter:
                kwargs["where"] = where_filter

            result = self.chroma_col.query(**kwargs)
            # result["ids"] là list[list[str]] (1 list per sub-question)
            ids: list[str] = []
            for id_list in result.get("ids", []):
                ids.extend(id_list)
            return ids

        chroma_ids = await loop.run_in_executor(None, _sync_query)
        print(f"   🔵 ChromaDB: {len(chroma_ids)} IDs từ {len(sub_queries)} sub-queries")
        return chroma_ids

    # ─────────────────────────────────────────
    # STEP B: FULL-TEXT SEARCH (PostgreSQL FTS)
    # ─────────────────────────────────────────
    async def _sql_fts_search(self, sub_queries: list[str], top_k: int) -> list[str]:
        """
        Tìm kiếm từ khóa bằng PostgreSQL Full-Text Search (thay thế BM25Retriever).

        Dataflow:
          sub_queries (list[str])
            → Ghép thành 1 chuỗi query (dùng | để OR giữa các từ)
            → asyncpg.Pool.fetch(SQL)
            → SQL: WHERE search_vector @@ plainto_tsquery('pg_catalog.simple', $1)
                   ORDER BY ts_rank(search_vector, ...) DESC
                   LIMIT top_k
            → Trả về list[str] chunk_ids đã sắp xếp theo ts_rank giảm dần
        """
        pool = await self._get_pg_pool()

        # Ghép tất cả sub-queries thành 1 chuỗi tìm kiếm
        combined_query = " ".join(sub_queries)

        sql = """
            SELECT chunk_id
            FROM document_chunks
            WHERE search_vector @@ plainto_tsquery('pg_catalog.simple', $1)
            ORDER BY ts_rank(search_vector, plainto_tsquery('pg_catalog.simple', $1)) DESC
            LIMIT $2
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, combined_query, top_k)

        sql_ids: list[str] = [row["chunk_id"] for row in rows]
        print(f"   🟢 PostgreSQL FTS: {len(sql_ids)} IDs")
        return sql_ids

    # ─────────────────────────────────────────
    # STEP C: LẤY TEXT THÔ TỪ POSTGRESQL
    # ─────────────────────────────────────────
    async def _fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[Document]:
        """
        Lấy content và metadata của các chunk theo danh sách chunk_id.

        Dataflow:
          chunk_ids (list[str])
            → asyncpg.Pool.fetch(SQL JOIN documents)
            → Trả về list[Document] với page_content=content, metadata={source, index, chunk_id}

        JOIN logic:
          document_chunks → documents
          → Lấy: content, chunk_index, file_name (cho metadata source)
        """
        if not chunk_ids:
            return []

        pool = await self._get_pg_pool()

        sql = """
            SELECT
                dc.chunk_id,
                dc.chunk_index,
                dc.content,
                d.file_name
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.chunk_id = ANY($1::text[])
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, chunk_ids)

        # Chuyển rows thành list[Document] để Jina Rerank xử lý
        docs: list[Document] = []
        for row in rows:
            docs.append(Document(
                page_content=row["content"],
                metadata={
                    "chunk_id": row["chunk_id"],
                    "source":   row["file_name"],   # Khớp với ChromaDB metadata "source"
                    "index":    row["chunk_index"],
                }
            ))
        return docs

    # ─────────────────────────────────────────
    # STEP D: LẤY CHAT HISTORY (SQLAlchemy ORM)
    # ─────────────────────────────────────────
    async def _fetch_chat_history(self, session_id: str, db: AsyncSession) -> str:
        """
        Lấy 10 tin nhắn gần nhất từ PostgreSQL theo session_id.
        Sử dụng SQLAlchemy AsyncSession (ORM) thay vì asyncpg raw SQL.

        Dataflow:
          session_id (str) + db (AsyncSession)
            → select(ChatHistory).where(session_id == ...).order_by(created_at DESC).limit(10)
            → Đảo ngược thứ tự (cũ → mới) để nạp vào context
            → Format: "user: ..." / "assistant: ..."
            → Ghép thành chuỗi context window

        Args:
            session_id: UUID của phiên chat
            db:         AsyncSession từ FastAPI Dependency (get_db)

        Returns:
            Chuỗi lịch sử chat đã format, hoặc thông báo "(Chưa có lịch sử hội thoại)"
        """
        # Import model ở đây để tránh circular import
        from backend.database import ChatHistory

        stmt = (
            select(ChatHistory)
            .where(ChatHistory.session_id == session_id)
            .order_by(desc(ChatHistory.created_at))
            .limit(CHAT_HISTORY_LIMIT)
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()

        if not rows:
            return "(Chưa có lịch sử hội thoại)"

        # Đảo ngược để hiển thị theo thứ tự thời gian (cũ → mới)
        history_lines = [f"{row.role}: {row.content}" for row in reversed(rows)]
        return "\n".join(history_lines)

    # ─────────────────────────────────────────
    # STEP E: LƯU CHAT HISTORY (SQLAlchemy ORM)
    # ─────────────────────────────────────────
    async def _save_chat_history(
        self,
        session_id: str,
        user_message: str,
        bot_response: str,
        db: AsyncSession,
        user_id: str | None = None
    ):
        """
        Lưu cặp (user_message, bot_response) vào bảng chat_history.
        Sử dụng SQLAlchemy AsyncSession (ORM) thay vì asyncpg raw SQL.
        Gọi sau khi streaming hoàn thành.

        Args:
            session_id:    UUID phiên chat
            user_message:  Câu hỏi của user
            bot_response:  Câu trả lời đầy đủ từ LLM (sau khi streaming hoàn tất)
            db:            AsyncSession từ FastAPI Dependency
            user_id:       UUID user đã đăng nhập (None nếu khách)
        """
        from backend.database import ChatHistory
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        db.add(ChatHistory(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=user_message,
            created_at=now
        ))
        db.add(ChatHistory(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=bot_response,
            created_at=now
        ))
        await db.commit()

    # ─────────────────────────────────────────
    # MAIN ENTRYPOINT: question()
    # ─────────────────────────────────────────
    async def question(
        self,
        cau_hoi: str,
        db: AsyncSession,
        session_id: str = "default",
        user_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Pipeline RAG đầy đủ từ câu hỏi → câu trả lời streaming.

        Dataflow tổng quát:
          cau_hoi
            → [A] AgentRouter (Gemini) → intent + source_filter + sub_queries (1 req)
            → [B1] ChromaDB query + Metadata Pre-filter (cosine) → chroma_ids
            → [B2] PostgreSQL FTS → sql_ids          (song song với B1)
            → [C] RRF gộp chroma_ids + sql_ids → merged_ids
            → [D] Fetch Text từ PostgreSQL theo merged_ids → docs (list[Document])
            → [E] Jina Rerank (top_n động) + Score Threshold 0.5 → final_docs
            → [F] Fetch Chat History (SQLAlchemy ORM) → history_str
            → [G] Format context + history → nạp vào LLM Groq (streaming)
            → yield token (từng ký tự streaming về FastAPI)
            → [H] Lưu (cau_hoi, full_answer) vào chat_history (SQLAlchemy ORM)

        Args:
            cau_hoi:    Câu hỏi của user
            db:         AsyncSession từ FastAPI Dependency (dùng cho chat_history ORM)
            session_id: ID phiên chat (từ request.session_id)
            user_id:    UUID của user đã đăng nhập (None nếu chưa đăng nhập)

        Yields:
            str: Từng token streaming từ LLM
        """
        print(f"\n{'='*50}")
        print(f"📥 Câu hỏi: {cau_hoi}")
        t_start = time.time()

        # ── [A] AGENT ROUTER (Gemini: Intent + Decompose + Metadata) ──
        print("🧭 [A] Agent Router (Gemini)...")
        route_result = await self.router.route(cau_hoi)
        sub_queries   = route_result["sub_queries"]
        source_filter = route_result["source_filter"]
        top_k         = route_result["top_k"]
        jina_top_n    = route_result["jina_top_n"]
        print(f"   Intent: {route_result['intent']} | top_k={top_k} | jina_top_n={jina_top_n}")

        # ── [B] HYBRID SEARCH SONG SONG ──
        print("🔍 [B] Hybrid Search (Chroma + PostgreSQL FTS)...")
        chroma_task = asyncio.create_task(
            self._chroma_search(sub_queries, top_k, source_filter=source_filter)
        )
        sql_task = asyncio.create_task(
            self._sql_fts_search(sub_queries, top_k)
        )
        chroma_ids, sql_ids = await asyncio.gather(chroma_task, sql_task)

        # ── [C] RRF: GỘP VÀ XẾP HẠNG ──
        print("⚖️  [C] Reciprocal Rank Fusion...")
        merged_ids = reciprocal_rank_fusion(chroma_ids, sql_ids)
        print(f"   Merged: {len(merged_ids)} IDs (unique)")

        # ── [D] LẤY TEXT THÔ TỪ POSTGRESQL ──
        print("📖 [D] Lấy text từ PostgreSQL...")
        docs = await self._fetch_chunks_by_ids(merged_ids)
        print(f"   Đã lấy {len(docs)} chunks")

        if not docs:
            print("⚠️  Không tìm thấy chunk nào liên quan. Trả lời không có ngữ cảnh.")
            context = "(Không tìm thấy tài liệu liên quan)"
        else:
            # ── [E] JINA RERANK + SCORE THRESHOLD ──
            print(f"🏆 [E] Jina Rerank → top {jina_top_n} (threshold={JINA_SCORE_THRESHOLD})...")
            self.jina.top_n = jina_top_n

            # compress_documents là synchronous → chạy trong executor
            loop = asyncio.get_running_loop()
            reranked_docs: list[Document] = await loop.run_in_executor(
                None,
                lambda: self.jina.compress_documents(documents=docs, query=cau_hoi)
            )

            # ── Score Threshold: Loại bỏ chunk kém liên quan ──
            final_docs: list[Document] = []
            for doc in reranked_docs:
                score = doc.metadata.get("relevance_score", 1.0)
                if score >= JINA_SCORE_THRESHOLD:
                    final_docs.append(doc)
                else:
                    print(f"   ⛔ Loại chunk '{doc.metadata.get('chunk_id','?')}' (score={score:.3f} < {JINA_SCORE_THRESHOLD})")

            print(f"   ✅ Sau threshold: {len(final_docs)}/{len(reranked_docs)} chunks được giữ lại")

            if not final_docs:
                # Trường hợp tất cả chunk đều dưới threshold → dùng câu trả lời không có ngữ cảnh
                context = "(Không tìm thấy tài liệu đủ liên quan)"
            else:
                # Format context để nạp vào prompt
                context_parts: list[str] = []
                for doc in final_docs:
                    chunk_id = doc.metadata.get("chunk_id", "?")
                    source   = doc.metadata.get("source",   "unknown")
                    index    = doc.metadata.get("index",    "?")
                    score    = doc.metadata.get("relevance_score", "?")
                    context_parts.append(
                        f"[{chunk_id}] source={source} | chunk={index} | relevance={score:.3f}\n"
                        f"{doc.page_content}"
                    )
                context = "\n\n---\n\n".join(context_parts)

        # ── [F] LẤY CHAT HISTORY (SQLAlchemy ORM) ──
        print("💬 [F] Lấy lịch sử chat (ORM)...")
        chat_history_str = await self._fetch_chat_history(session_id, db)

        t_search = time.time()
        print(f"⏱️  Tổng thời gian tìm kiếm: {t_search - t_start:.2f}s")

        # ── [G] STREAMING TỪ LLM GROQ ──
        print("🤖 [G] Streaming từ LLM Groq...\n")
        full_response_parts: list[str] = []
        async for token in self.deploy.astream({
            "lich_su":  chat_history_str,
            "ngu_canh": context,
            "cau_hoi":  cau_hoi
        }):
            chunk_text: str = token.content if hasattr(token, "content") else str(token)
            full_response_parts.append(chunk_text)
            yield chunk_text

        # ── [H] LƯU CHAT HISTORY (SQLAlchemy ORM) ──
        full_response = "".join(full_response_parts)
        await self._save_chat_history(
            session_id=session_id,
            user_message=cau_hoi,
            bot_response=full_response,
            db=db,
            user_id=user_id
        )
        t_end = time.time()
        print(f"\n⏱️  Tổng thời gian end-to-end: {t_end - t_start:.2f}s")


# ─────────────────────────────────────────────
# TEST LOCAL
# ─────────────────────────────────────────────
async def _run_test():
    """Test nội bộ - cần mock AsyncSession khi chạy không qua FastAPI."""
    from unittest.mock import AsyncMock, MagicMock

    # Mock AsyncSession để test cục bộ mà không cần kết nối DB thật
    mock_db = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result

    bot = BotAi()
    question = "Dựa trên các câu hỏi trên, bạn nghĩ tôi là 1 người đi theo lĩnh vực nào?"
    print("Testing llm_client.py locally...\n")
    async for chunk in bot.question(
        cau_hoi=question,
        db=mock_db,
        session_id="test_session_001"
    ):
        print(chunk, end="", flush=True)
    await bot.close()

if __name__ == "__main__":
    asyncio.run(_run_test())