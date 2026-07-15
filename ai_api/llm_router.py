"""
LLM ROUTER - Bộ định tuyến thông minh cho hệ thống RAG
=======================================================
Mục tiêu: Gom chung 3 nhiệm vụ tốn API vào 1 lần gọi Gemini để bảo toàn quota Groq:
  1. Intent Detection:  "summarize" (cần nhiều chunk) vs "detail" (cần ít chunk)
  2. Metadata Extraction: Bóc tách tên file (.pdf) hoặc URL từ câu hỏi → làm Where Filter ChromaDB
  3. Query Decomposition: Tách câu hỏi gốc thành 3 sub-queries ngắn gọn để Hybrid Search

Luồng dữ liệu:
  user_query (str)
    → Gemini API (1 request duy nhất)
    → JSON: { intent, source_filter, sub_queries }
    → BotAi.question() dùng để điều hướng pipeline

Tại sao dùng Gemini thay Groq ở bước này?
  - Groq (free): 30 Request/phút → phải bảo toàn cho bước trả lời cuối cùng
  - Gemini (free): 1500 Request/ngày, 15 RPM → dư sức phục vụ routing
  - Tách biệt trách nhiệm: Gemini = Router, Groq = Generator
"""

import os, re, json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Config top_k cho từng intent:
# - summarize: cần ngữ cảnh rộng (20 chunks) → hỏi ChromaDB nhiều hơn
# - detail:    cần ngữ cảnh sâu, cụ thể (2-3 chunks) → chỉ lấy top chunk chính xác nhất
INTENT_CONFIG = {
    "summarize": {"top_k": 40, "jina_top_n": 20},
    "detail":    {"top_k": 10, "jina_top_n": 3},
}
DEFAULT_INTENT = "detail"  # Fallback nếu Gemini không phân loại được


# ─────────────────────────────────────────────
# ROUTER PROMPT
# ─────────────────────────────────────────────
_ROUTER_PROMPT = """\
Bạn là hệ thống phân tích câu hỏi của hệ thống RAG (Retrieval-Augmented Generation).
Nhiệm vụ: Phân tích câu hỏi sau và trả về JSON THUẦN TÚY (không có markdown, không có ```json).

PHÂN LOẠI INTENT:
- "summarize": Khi user muốn TÓM TẮT, TỔNG HỢP, LIỆT KÊ NHIỀU THÔNG TIN
  Ví dụ: "tóm tắt", "tổng hợp", "liệt kê tất cả", "có những gì", "overview", "summary", "nêu các", "trình bày"
- "detail": Khi user muốn GIẢI THÍCH SÂU, ĐỊNH NGHĨA, SO SÁNH, PHÂN TÍCH CỤ THỂ MỘT ĐỐI TƯỢNG
  Ví dụ: "định nghĩa", "tại sao", "cách thức", "giải thích", "ý nghĩa là gì", "so sánh X và Y"

TRÍCH XUẤT SOURCE FILTER:
- Nếu câu hỏi đề cập đến tên file (VD: "baocao.pdf", "slide_chuong1.pdf"), trích xuất tên file đó.
- Nếu câu hỏi đề cập đến URL (VD: "https://..."), trích xuất URL đó.
- Nếu không có, đặt source_filter là null.

PHÂN RÃ CÂU HỎI (sub_queries):
- Tách câu hỏi thành ĐÚNG 3 câu hỏi phụ ngắn gọn (3-8 từ mỗi câu) để tìm kiếm tài liệu.
- Mỗi câu phụ phải bao phủ một khía cạnh khác nhau của câu hỏi gốc.
- Dùng ngôn ngữ đơn giản, tránh từ hỏi phức tạp.

Câu hỏi gốc: {cau_hoi}

Trả về JSON theo đúng cấu trúc này (không thêm bất kỳ ký tự nào khác):
{{
  "intent": "detail hoặc summarize",
  "source_filter": "tên_file.pdf hoặc https://... hoặc null",
  "sub_queries": ["câu hỏi phụ 1", "câu hỏi phụ 2", "câu hỏi phụ 3"]
}}
"""


# ─────────────────────────────────────────────
# AGENT ROUTER CLASS
# ─────────────────────────────────────────────
class AgentRouter:
    """
    Bộ định tuyến thông minh sử dụng Gemini API.

    Attributes:
        llm:      ChatGoogleGenerativeAI - Gemini Flash (nhanh, giới hạn request cao)
        chain:    PromptTemplate | llm - chain gọi Gemini
    """

    def __init__(self):
        # Dùng gemini-2.0-flash: siêu nhanh, free tier 15 RPM / 1500 req/ngày
        # Không dùng gemini-pro để tiết kiệm quota và giảm latency
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,         # Temperature=0 để output JSON ổn định nhất
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        self.prompt = PromptTemplate.from_template(_ROUTER_PROMPT)
        self.chain = self.prompt | self.llm

    async def route(self, cau_hoi: str) -> dict:
        """
        Phân tích câu hỏi và trả về cấu trúc routing.

        Args:
            cau_hoi: Câu hỏi gốc từ user

        Returns:
            dict với các trường:
              - intent (str):         "summarize" hoặc "detail"
              - source_filter (str|None): Tên file/URL nếu user đề cập, ngược lại None
              - sub_queries (list[str]): Danh sách 3 sub-queries để Hybrid Search
              - top_k (int):          Số kết quả lấy từ mỗi nguồn tìm kiếm
              - jina_top_n (int):     Số chunk cuối cùng sau Jina Rerank

        Raises:
            Không raise exception - luôn có giá trị fallback an toàn
        """
        try:
            result = await self.chain.ainvoke({"cau_hoi": cau_hoi})
            raw_text: str = result.content if hasattr(result, "content") else str(result)

            # Xử lý trường hợp Gemini bọc trong ```json ... ```
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)

            parsed: dict = json.loads(raw_text.strip())

            # Validate và lấy intent hợp lệ
            intent = parsed.get("intent", DEFAULT_INTENT).lower()
            if intent not in INTENT_CONFIG:
                print(f"⚠️ Router: Intent '{intent}' không hợp lệ, dùng fallback '{DEFAULT_INTENT}'")
                intent = DEFAULT_INTENT

            # Lấy source_filter (None nếu "null" hoặc không có)
            source_filter = parsed.get("source_filter")
            if source_filter == "null" or not source_filter:
                source_filter = None

            # Lấy sub_queries, fallback về câu gốc nếu lỗi
            sub_queries: list[str] = parsed.get("sub_queries", [])
            if not sub_queries or not isinstance(sub_queries, list):
                print("⚠️ Router: Không parse được sub_queries, dùng câu gốc làm fallback.")
                sub_queries = [cau_hoi]
            sub_queries = [q.strip() for q in sub_queries[:3] if q.strip()]

            config = INTENT_CONFIG[intent]

            print(f"   🧭 Router → Intent: {intent} | Source: {source_filter} | Sub-queries: {sub_queries}")
            return {
                "intent":        intent,
                "source_filter": source_filter,
                "sub_queries":   sub_queries,
                "top_k":         config["top_k"],
                "jina_top_n":    config["jina_top_n"],
            }

        except (json.JSONDecodeError, Exception) as e:
            # Fallback an toàn: dùng câu gốc, không filter source
            print(f"⚠️ Router Error ({type(e).__name__}: {e}). Dùng fallback toàn bộ.")
            config = INTENT_CONFIG[DEFAULT_INTENT]
            return {
                "intent":        DEFAULT_INTENT,
                "source_filter": None,
                "sub_queries":   [cau_hoi],
                "top_k":         config["top_k"],
                "jina_top_n":    config["jina_top_n"],
            }
