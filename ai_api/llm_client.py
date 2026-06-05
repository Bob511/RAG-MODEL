from langchain_core.prompts import PromptTemplate # tạo langchain nhưng chỉ lấy core và phần prompts (để hạn chế ô nhớ và tối ưu tốc độ)
import time, os
from langchain_core.documents import Document
from typing import Generator, List
# 1. Cập nhật cách gọi Chroma theo tiêu chuẩn gói độc lập
from langchain_chroma import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever, BM25Retriever
from langchain_community.document_compressors import JinaRerank
from langchain_groq import ChatGroq
from check_ultis import check_database
from ingestion_processing import read_and_write_chromaDB
from dotenv import load_dotenv
load_dotenv()
# thử nghiệm xem chromaDB (hay database đã chạy chưa)
if not check_database():
    raise ValueError ("Database still unavailable, please try again!")

class BotAi:
    def __init__(self):
       # callAPI từ groq với LOGIC: cần apikey, cần tên model, cần url (nếu không sử dụng thư viện), thêm nhiệt độ (nếu cần)
       # Cần 1 promptTemplate để làm khuôn (LLMs hiểu dễ hơn)
        groq_ai = os.getenv("API_KEY_GROQ")
        self.llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7, api_key=groq_ai)

        prompt = '''Bạn là trợ lý AI, chuyên phục vụ cho việc tìm hiểu, phân tích các thông tin từ file PDF (nếu có) và đưa ra câu trả lời 
        dựa vào những gì bạn đã được cung cấp. Yêu cầu đưa ra ngôn ngữ trả lời phụ thuộc vào câu hỏi của user và ghi chú nguồn đã lấy trên database (Ví dụ: ID_TÀI_LIỆU - TÊN FILE: 01-ĐẠI SỐ TUYẾN TÍNH)
        Ngữ cảnh của file PDF: {ngu_canh}
        
        Câu hỏi của user: {cau_hoi}
        Trả lời: '''
        self.template = PromptTemplate(
            input_variables= ["ngu_canh", "cau_hoi"],
            template=prompt
        )
        self.deploy = self.template | self.llm # tích hợp template vào chuỗi llms
        self.hybrid_retrievers = None
    def stream_text(self, file_path : str) -> Generator[Document, None, None]:
        # TESTING WITH FILE AVAILABLE ONLY IN HARD DISK
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Mô hình Llama-4 mã nguồn mở có hiệu năng vượt trội.\n")
                f.write("Hệ thống RAG kết hợp ChromaDB và BM25 tối ưu độ chính xác.\n")
                f.write("FastAPI được sử dụng để xây dựng hệ thống API Gateway cốt lõi.\n")
        with open(file_path, "r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                clean = line.strip()
                if clean:
                    yield Document(page_content=line, metadata={"source" : file_path, "line" : index})
    def hybrid_search(self, file_path):
        doc_list = list(self.stream_text(file_path))
        bm25_retrievers = BM25Retriever.from_documents(doc_list)
        bm25_retrievers.k = 10
        chroma_vectorstore = Chroma.from_documents(doc_list, embedding=None) 
        chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": bm25_retrievers.k})
        # 3. Hợp nhất hai bộ truy xuất chạy song song bằng EnsembleRetriever
        print(" Bước 4: Hợp nhất luồng truy xuất bằng thuật toán lai (Hybrid Search)...")
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retrievers, chroma_retriever],
            weights=[0.5, 0.5] # Phân bổ trọng số cân bằng 50% từ khóa - 50% ngữ nghĩa
        )
        jina_compress = JinaRerank(jina_api_key=os.getenv("JINA_API"), top_n=3)
        compress_retriever = ContextualCompressionRetriever(base_compressor=jina_compress, base_retriever=ensemble)
        return compress_retriever
    def question(self, ngu_canh : str, cau_hoi : str, file_path : str) -> dict:
        print("Trả lời...")
        begin = time.time()
        if not self.hybrid_retrievers:
            self.hybrid_retrievers = self.hybrid_search(file_path=file_path)
        relevant_docs = self.hybrid_retrievers.ainvoke(cau_hoi)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        stop = time.time()
        track_time = stop - begin
        print("Thời gian tìm kiếm và trả kết quả: ", track_time)
        begin = time.time()
        #invoke để kích hoạt và chạy ai (deploy chạy ai nhờ vào | ở trước)
        result = self.deploy.ainvoke({
            "ngu_canh" : context,
            "cau_hoi": cau_hoi
            # hiển thị ra (vd: human: tôi là Dân; AI: chào Dân)
        })
        stop = time.time()
        track_time = stop - begin
        print(f"----- time to run AI is: {track_time:.4f}s -----")
        return result

if __name__ == '__main__':
    test = BotAi()
    FILE = os.getenv("FILE_NAME")
    CHROMA_HOST = os.getenv("CHROMA_CONTAINER_NAME")
    question = "tất cả những gì tôi cần biết về tài liệu đã được cung cấp" # Đây là nơi bạn đặt câu hỏi 
    situation = read_and_write_chromaDB(file=FILE, host_chroma=CHROMA_HOST) # Sử dụng chức năng đọc và phân tích vector nhúng từ thành viên 1
    ask = situation.ask_ans(question)
# Đây là ngữ cảnh của prompt (có thể tạo nhiều situation để chạy nhiều lần test AI)
    dap_an = test.question(ngu_canh=ask, cau_hoi=question, file_path=FILE) #Bắt đầu chạy AI theo lần lượt ngữ cảnh và câu hỏi
    print(dap_an)