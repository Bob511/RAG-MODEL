import re, chromadb, os
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from dotenv import load_dotenv
import ollama
load_dotenv()

# ================================================================== #
#  CONFIG                                                             #
# ================================================================== #

HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

CHUNK_SIZE_MAP = {
    "QUIZ":    800,
    "LEGAL":   1000,
    "FAQ":     1000,
    "DEFAULT": 1000,
}

# Bảng được xử lý riêng — dùng chunk_size lớn hơn bình thường
TABLE_CHUNK_SIZE   = 2000   # Tối đa ký tự mỗi table-chunk
TABLE_ROWS_PER_SUB = 15     # Nếu bảng quá lớn, chia mỗi sub-chunk N dòng data

# ================================================================== #
#  1. PHÂN LOẠI TÀI LIỆU                                             #
# ================================================================== #

# Dùng class với kế thừa từ BaseModel để tiến hành gán vào model gemini thông qua API
class DocumentClassification(BaseModel):
    doc_type: str = Field(
        description="""Phân loại tài liệu thành 1 trong 4 danh mục:
        - 'QUIZ': Đề thi trắc nghiệm, bài tập có câu hỏi và đáp án A, B, C, D.
        - 'LEGAL': Văn bản pháp lý, tiêu chuẩn quốc tế (ISO), hợp đồng, quy chế.
        - 'FAQ': Tài liệu hỏi đáp kỹ thuật, hướng dẫn từng bước.
        - 'DEFAULT': Văn bản đoạn văn thông thường không thuộc 3 loại trên.
        """
    )
    author: str = Field(
        description="Tên tác giả, nhóm tác giả hoặc cơ quan ban hành tài liệu. Trả về chuỗi rỗng '' nếu không tìm thấy."
    )
    website: str = Field(
        description="Tên tổ chức, domain, URL hoặc tên website cung cấp/sản xuất tài liệu. Trả về chuỗi rỗng '' nếu không tìm thấy."
    )

def llm_classify_document(sample_text: str) -> str:
    '''LOGIC: Kết nối tới gemini 2.5 flash thông qua API -> Sau đó gán phần description thông qua class ở trên -> tạo prompt và dùng invoke() để tiến hành chạy'''
    print("🤖 Đang phân loại tài liệu bằng Gemini...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    structured_llm = llm.with_structured_output(DocumentClassification)
    prompt = f"""
    Bạn là chuyên gia phân tích tài liệu. Đọc đoạn văn bản sau (thường là trang bìa hoặc phần giới thiệu của tài liệu).
    Hãy xác định:
    1. Loại tài liệu (doc_type)
    2. Tác giả (author)
    3. Website/Nguồn gốc (website)
    
    Văn bản mẫu:
    ---
    {sample_text}
    ---
    """
    try:
        result = structured_llm.invoke(prompt)
        return result
    except Exception as e:
        print(f"⚠️ Lỗi trích xuất LLM: {e}")
        # Trả về giá trị mặc định nếu LLM lỗi
        return DocumentClassification(doc_type="DEFAULT", author="", website="")



'''LOGIC:
1. re.compile() kiểm tra docs dạng bảng
2. ?: áp dụng cho cả dòng
3. [ \t]*\| : thụt lề dòng là có thể và bắt đầu bằng dấu |
4. [^\n]+ : dấu + kiểm tra xem bên trong có các ký tự không, dấu ^\n biểu thị có bất kỳ khoảng trắng?
5. \n là xuống dòng
6. {2,} thỏa 2 dòng liên tiếp đảm bảo là dạng bảng '''
_TABLE_BLOCK_RE = re.compile(
    r'(?is)(<table\b[^>]*>.*?</table>|(?:[ \t]*\|[^\n]+(?:\n|$)){2,})'
)
def _segment_text_and_tables(text: str) -> List[Tuple[str, str]]:
    """
    LOGIC: kiểm tra dạng bảng hoặc docs (thông qua dòng 85)
    1. dùng for để kiểm tra lần lượt các dòng của text
    2. vì dòng 85 là thuộc tính của class re -> bản thân nó cx có các thuộc tính con riêng -> dùng start() và end() để lấy index
    3. Tuy nhiên, nếu trong text có bảng nhưng bảng ở mid (còn phần start và end) là text, ta sẽ tiến hành gán text cho chúng
    4. Thông qua index ở start và end() phía trên, ta lấy biến tạm và chạy index từ biến tạm -> start và end sẽ là biến tạm đến hết (sau khi xong for)
    """
    segments: List[Tuple[str, str]] = []
    last_end = 0

    for match in _TABLE_BLOCK_RE.finditer(text):
        start, end = match.start(), match.end()

        # Phần TEXT trước bảng
        before = text[last_end:start]
        if before.strip():
            segments.append(("TEXT", before))

        # Bảng
        table_text = match.group(0)
        if table_text.strip():
            segments.append(("TABLE", table_text))

        last_end = end

    # Phần TEXT còn lại sau bảng cuối
    tail = text[last_end:]
    if tail.strip():
        segments.append(("TEXT", tail))

    return segments


def _parse_table_rows(table_text: str) -> Tuple[List[str], List[str]]:
    """
    LOGIC: Mỗi khi tách chunk table, nếu table quá dài, ta tách ra sub-chunks, tuy nhiên nếu ko có header, AI sẽ không biết chunk table này có nội dung là ntn có hợp với prompt ko
    1. Tiến hành tách các dòng chunks table thành 1 list để ktra độ dài Nếu nhỏ hơn 2 dòng, trả về vì ko phải là table
    2. lần lượt kiểm tra xem header ở đâu
    3. Trả về index và gán index vào text để láy header và data
    """
    lines = [ln for ln in table_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return lines, []

    # Dòng separator là dòng chứa |---| hoặc |===|
    sep_idx = None
    '''[\s\-\|:=]+ : lần lượt kiểm tra các dấu - : = | và các ký tự bên trong để kiểm tra dòng này là header? '''
    for i, ln in enumerate(lines):
        if re.match(r'[ \t]*\|[\s\-\|:=]+\|', ln):
            sep_idx = i
            break

    if sep_idx is None:
        # Không tìm thấy separator → coi toàn bộ là data
        return [], lines

    header_lines = lines[:sep_idx + 1]   # header + separator
    data_lines   = lines[sep_idx + 1:]   # các dòng data
    return header_lines, data_lines


def chunk_table(table_text: str, base_metadata: dict) -> List[Document]:
    table_text = table_text.strip()

    # --- NHÁNH 1: XỬ LÝ BẢNG HTML ---
    if table_text.lower().startswith("<table"):
        # Với HTML, tốt nhất là giữ nguyên toàn vẹn cấu trúc thẻ.
        # Nếu bảng HTML quá lớn vượt token của LLM, bạn sẽ cần dùng BeautifulSoup 
        # để bóc tách <thead> và <tbody> riêng (phức tạp hơn).
        # Tạm thời ở mức độ này, ta gói trọn nó vào 1 chunk.
        meta = {**base_metadata, "content_type": "table_html", "row_range": "all"}
        return [Document(page_content=table_text, metadata=meta)]

    # --- NHÁNH 2: XỬ LÝ BẢNG MARKDOWN (Code cũ của bạn) ---
    header_lines, data_lines = _parse_table_rows(table_text)
    header_str = "\n".join(header_lines) + "\n" if header_lines else ""

    if len(table_text) <= TABLE_CHUNK_SIZE:
        meta = {**base_metadata, "content_type": "table_md", "row_range": "all"}
        return [Document(page_content=table_text, metadata=meta)]

    chunks: List[Document] = []
    for i in range(0, max(len(data_lines), 1), TABLE_ROWS_PER_SUB):
        row_group  = data_lines[i : i + TABLE_ROWS_PER_SUB]
        chunk_text = header_str + "\n".join(row_group)
        row_start  = i + 1
        row_end    = min(i + TABLE_ROWS_PER_SUB, len(data_lines))
        meta = {
            **base_metadata,
            "content_type": "table_md",
            "row_range": f"{row_start}-{row_end}",
        }
        chunks.append(Document(page_content=chunk_text.strip(), metadata=meta))

    return chunks


# ================================================================== #
#  4. TEXT PIPELINE: 2-pass (header split → secondary split)         #
# ================================================================== #

def _split_by_headers(text: str) -> List[Document]:
    # tách chunk theo từng header từ markdown
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
        return_each_line=False,
    )
    return splitter.split_text(text)


def _get_secondary_splitter(doc_type: str, chunk_size: int):
    '''LOGIC: Tách các chunk nhỏ hơn (cho text type sau khi phần header ở trên)
    1. phân chia từng loại docs -> Với từng loại doc_type ta có từng loại chia nhỏ khác nhau
    2. sử dụng RecursiveCharacterTextSplitter()
    3. Với dạng Default -> dùng Embedding của gemini và semanticchunker để phân chia thông minh '''
    overlap = max(50, chunk_size // 20)

    if doc_type == "QUIZ":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[r"\nCâu \d+", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
            is_separator_regex=True,
        )
    elif doc_type == "LEGAL":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                r"\nĐiều \d+", r"\nChương \d+",
                r"\n\d+\.\d+[\. ]",
                "\n\n", "\n- ", "\n", ". ", " ", ""
            ],
            keep_separator=True,
            is_separator_regex=True,
        )
    else:
        api_key = os.getenv("Mistral_key")
        embeddings = OllamaEmbeddings(model='qwen3-embedding:0.6b')
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )

def _count_sentences(text: str) -> int:
    """Đếm số câu trong text theo dấu câu cơ bản."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([s for s in sentences if s.strip()])

def _process_text_segment(
    text: str,
    doc_type: str,
    chunk_size: int,
    secondary_splitter,
) -> List[Document]:
    """
    2-pass pipeline cho đoạn TEXT thuần:
      Pass 1: MarkdownHeaderTextSplitter
      Pass 2: secondary_splitter nếu chunk còn quá lớn
    """
    header_chunks = _split_by_headers(text)
    result: List[Document] = []

    for chunk in header_chunks:
        meta = {**chunk.metadata, "content_type": "text"}
        content = chunk.page_content
        if len(chunk.page_content) <= chunk_size:
            result.append(Document(page_content=chunk.page_content, metadata=meta))
        else:
            is_semantic = isinstance(secondary_splitter, SemanticChunker)
            too_few_sentences = is_semantic and _count_sentences(content) < 2

            if too_few_sentences:
                # Chỉ 1 câu nhưng vượt chunk_size (hiếm) → giữ nguyên, không crash
                result.append(Document(page_content=content, metadata=meta))
            else:
                try:
                    sub_texts = secondary_splitter.split_text(content)
                    for t in sub_texts:
                        result.append(Document(page_content=t, metadata=meta))
                except IndexError:
                    # Fallback an toàn nếu vẫn xảy ra edge case
                    print(f"⚠️ SemanticChunker lỗi với chunk {len(content)} ký tự → giữ nguyên")
                    result.append(Document(page_content=content, metadata=meta))

    return result


# ================================================================== #
#  5. HÀM CHUNKING CHÍNH                                             #
# ================================================================== #

def adaptive_chunking_md(md_path: str) -> List[Document]:
    path = Path(md_path)
    if not path.exists() or path.suffix.lower() != ".md":
        raise ValueError(f"File không hợp lệ: {md_path}")

    print(f"\n📄 Đang đọc file: {md_path}")
    text = path.read_text(encoding="utf-8")

    if not text.strip():
        print("⚠️ File rỗng.")
        return []

    # --- Phân loại ---
    sample   = text[:1500]
    extracted_meta = llm_classify_document(sample) # Gọi hàm mới
    doc_type = extracted_meta.doc_type
    author = extracted_meta.author
    website = extracted_meta.website
    print(f"🎯 Loại tài liệu: [{doc_type}]")
    text = clean_markdown_page_breaks(text)
    chunk_size         = CHUNK_SIZE_MAP.get(doc_type, 1000)
    secondary_splitter = _get_secondary_splitter(doc_type, chunk_size)

    # --- Pre-segmentation: tách TABLE và TEXT ---
    print("🔍 Pre-segmentation: phát hiện bảng Markdown...")
    segments = _segment_text_and_tables(text)

    n_tables = sum(1 for t, _ in segments if t == "TABLE")
    n_texts  = sum(1 for t, _ in segments if t == "TEXT")
    print(f"   → {n_texts} đoạn TEXT | {n_tables} đoạn TABLE")

    # --- Xử lý từng segment ---
    final_chunks: List[Document] = []

    for seg_type, seg_content in segments:
        if seg_type == "TABLE":
            table_chunks = chunk_table(seg_content, base_metadata={"doc_type": doc_type})
            final_chunks.extend(table_chunks)

        else:  # TEXT
            text_chunks = _process_text_segment(
                seg_content, doc_type, chunk_size, secondary_splitter
            )
            final_chunks.extend(text_chunks)

    print(f"✅ Tổng cộng {len(final_chunks)} chunks (loại: {doc_type})")
    return push_to_chroma(final_chunks,file_path= path, author= author, website= website)

import re

def clean_markdown_page_breaks(text: str) -> str:
    """
    Xóa bỏ các dòng chỉ chứa dấu `---` (thường là page break của LlamaParse).
    Sử dụng cờ re.MULTILINE để neo dấu ^ và $ vào đầu/cuối của từng dòng.
    """
    # Pattern giải thích:
    # ^ : Bắt đầu dòng
    # \s* : Có thể có khoảng trắng
    # --- : Đúng 3 dấu gạch ngang (hoặc bạn có thể dùng -{3,} nếu có nhiều hơn 3 dấu)
    # \s* : Có thể có khoảng trắng
    # $ : Kết thúc dòng
    
    clean_text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Tùy chọn: Xóa bớt các khoảng trống (dòng trống) thừa thãi bị sinh ra sau khi xóa
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    
    return clean_text

def push_to_chroma(final_chunks, file_path: str, author: str = "", website: str = ""):
    # ... (Phần code kết nối ChromaDB HttpClient giữ nguyên như trên) ...
    
    # --- XỬ LÝ TÊN FILE VÀ TÁC GIẢ/WEBSITE Ở ĐÂY ---
    # 1. Lấy tên file gốc
    file_name_full = os.path.basename(file_path)             # VD: Danyeus.pdf
    file_name_no_ext = os.path.splitext(file_name_full)[0]   # VD: Danyeus
    
    # 2. Logic: Có tác giả thì dùng tác giả, không có thì dùng website
    source_origin = author.strip() if author.strip() else website.strip()

    docs = []
    metas = []
    ids = []
    embeddings=[]
    for i, chunk in enumerate(final_chunks):
        # ==========================================
        # YÊU CẦU 1: ID (Cú pháp: ten file goc _ stt)
        # ==========================================
        chunk_id = f"{file_name_no_ext}_{i+1}" 
        # (Ví dụ kết quả: Danyeus_1, Danyeus_2...)
        
        # ==========================================
        # YÊU CẦU 2: METADATA
        # ==========================================
        safe_meta = {}
                # Gọi Ollama chạy local để lấy vector
        response = ollama.embed(
            model='qwen3-embedding:0.6b', 
            input=chunk.page_content
                )
        vector = response['embeddings'][0]       
        embeddings.append(vector)
                
        # A. Giữ lại các metadata xịn từ bước chunking (như Header, Table)
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                safe_meta[key] = value
            else:
                safe_meta[key] = str(value)
                
        # B. Thêm Tên file gốc
        safe_meta["file_name"] = file_name_full
        
        # C. Thêm Tác giả hoặc Website (nếu có truyền vào)
        if source_origin:
            safe_meta["source"] = source_origin

        # Đẩy vào mảng
        docs.append(chunk.page_content)
        metas.append(safe_meta)
        ids.append(chunk_id)
        ChromaAPI = os.getenv("ChromaAPI")
        client = chromadb.CloudClient(
            api_key= ChromaAPI,
            tenant='4798bb4f-8541-44e6-ab6f-6b6594fcef7a',
            database='BIZRAG'
            )
        COLLECTION_NAME = 'Bigchild'
        # 2. Tạo hoặc lấy Collection trên Cloud
        collection = client.get_or_create_collection(name = COLLECTION_NAME)
        collection.upsert(
                    ids=ids,
                    documents=docs,
                    metadatas=metas,
                    embeddings=embeddings
                )
    # ... (Gọi collect.upsert như cũ) ...

if __name__ == "__main__":
    BASE_DIR = Path(r"D:\ragmodel\data_rag_output\report_cleaned.md")
    md_file  = BASE_DIR 
    chunks = adaptive_chunking_md(md_file)
    with open(r"D:\ragmodel\data_rag_output\data_test2.md", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"--- Chunk {i} ---\n")
            f.write(f"Metadata: {chunk.metadata}\n")
            f.write(f"{chunk.page_content}\n\n")
    # Thống kê phân bố loại chunk
    text_chunks  = [c for c in chunks if c.metadata.get("content_type") == "text"]
    table_chunks = [c for c in chunks if c.metadata.get("content_type") == "table"]
    print(f"\n📊 Phân bố: {len(text_chunks)} text chunks | {len(table_chunks)} table chunks")