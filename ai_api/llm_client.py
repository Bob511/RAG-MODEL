''' MỤC TIÊU CHÍNH: khởi tạo LLMs và tìm kiếm các thông tin trên database bằng hybrid search. 
Với input: user prompt, file pdf của user (nếu có). 
Với output: JSON với nội dung chính là câu trả lời của LLMs
DATAFLOW: FastAPI -> LLm_client.py -> nhúng câu hỏi user -> tìm thông tin liên quan trong database với hybrid search (BM25 - Ensemble) 
-> chọn lọc các thông tin chunk có liên quan nhất bằng Jina -> Trả ra kết quả database -> đưa vào LLMs với prompt input(user pronpt + thông tin liên quan)
-> Trả ra kết quả cuối cùng là dạng JSON'''
from langchain_core.prompts import PromptTemplate # tạo langchain nhưng chỉ lấy core và phần prompts (để hạn chế ô nhớ và tối ưu tốc độ)
import time, os, asyncio, chromadb
from langchain_core.documents import Document
from typing import Generator
# 1. Cập nhật cách gọi Chroma theo tiêu chuẩn gói độc lập
from langchain_chroma import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever, BM25Retriever
from langchain_community.document_compressors import JinaRerank
from langchain_groq import ChatGroq
from check_ultis import check_database
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
        self.llm = ChatGroq(model=os.getenv("MODEL"), temperature=0.7, api_key=groq_ai)

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

        prompt_decompose = '''Bạn là hệ thống tách câu hỏi. Chia câu hỏi của sau thành 3 câu hỏi phụ đơn giản. ngắn gọn hơn để tìm kiếm tài liệu.
        Tuyệt đối chỉ xuất ra câu hỏi, mỗi câu hỏi nằm trên 1 dòng, bắt đầu bằng dấu gạch ngang (-). Không giải thích.
        Câu hỏi gốc: {cau_hoi}'''
        self.decompose = PromptTemplate.from_template(prompt_decompose) | self.llm # hàm ainvoke() và invoke() chỉ có trong class của promptTemplate -> cần tích hợp template với model để chạy
        self.hybrid_retrievers = None

    def stream_text(self, file_path : str) -> Generator[Document, None, None]:
        # Mục đích: cấp data cho BM25
        access_cloud = chromadb.CloudClient(tenant=os.getenv("TENANT"), api_key= os.getenv("CHROMA_API"), database= "VectorManagement")
        access_database = access_cloud.get_collection(name="VectorChunk")
        get_docs = access_database.get(include=['documents', 'metadatas'])
        doc_list = []
        for doc_text, meta in zip(get_docs['documents'], get_docs['metadatas']):
            doc_list.append(Document(page_content=doc_text, metadata= meta))
        return doc_list
    
    def hybrid_search(self, file_path):
        doc = list(self.stream_text(file_path))
        bm25_retrievers = BM25Retriever.from_documents(doc)
        bm25_retrievers.k = 10
        
        chroma_vectorstore = Chroma.from_documents(doc, embedding=None) 
        chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": bm25_retrievers.k})
        # 3. Hợp nhất hai bộ truy xuất chạy song song bằng EnsembleRetriever
        print(" Bước 4: Hợp nhất luồng truy xuất bằng thuật toán lai (Hybrid Search)...")
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retrievers, chroma_retriever],
            weights=[0.5, 0.5] # Phân bổ trọng số cân bằng 50% từ khóa - 50% ngữ nghĩa
        )
        jina_compress = JinaRerank(jina_api_key=os.getenv("JINA_API"), top_n=3)
        self.hybrid_retrievers = ContextualCompressionRetriever(base_compressor=jina_compress, base_retriever=ensemble)
    
    async def question(self, cau_hoi : str, file_path : str) -> dict:
        print("Trả lời...")
        begin = time.time()
        if not self.hybrid_retrievers:
            self.hybrid_search(file_path=file_path)
        sub_questions = await self.decompose.ainvoke({"cau_hoi": cau_hoi}) 
        # Cắt chuỗi thành mảng các câu hỏi phụ - prompt có nếu ra - và \n -> thay thế - và dùng \n tách thành 1 đoạn
        sub_queries = [q.replace("-", "").strip() for q in sub_questions.content.split('\n') if q.strip()]
        print([doc for doc in sub_queries])
        # chạy for để tìm kiếm thông tin trên database thông qua hybrid search (BM25 + Chroma)
        store_relevant_docs = []
        for i in sub_queries:
            relevant_docs = await self.hybrid_retrievers.base_retriever.ainvoke(i) # sử dụng await cho ainvoke() - Document type() vì đây là search từ database nên cần document lưu trữ docs và metadata
            store_relevant_docs.extend(relevant_docs) # extend() dùng để thêm từng phần tử trong list, tuple vào list. Tức là thay vì thêm 1 mục (list hoặc tuple) thì đây sẽ là thêm từng phần tử
        unique_docs = []
        seen = set()
        # lọc chunk trùng
        for i in store_relevant_docs:
            if i not in seen:  
                seen.add(i)
                unique_docs.append(i)
        # đánh giá bằng jina rerank
        final_docs = self.hybrid_retrievers.base_compressor.compress_documents(documents=unique_docs, query=cau_hoi)
        # ngữ cảnh
        context = "\n".join([doc.page_content for doc in final_docs])
        print([doc for doc in context])
        stop = time.time()
        track_time = stop - begin
        print("Thời gian tìm kiếm và trả kết quả: ", track_time)
        begin = time.time()
        #invoke để kích hoạt và chạy ai (deploy chạy ai nhờ vào | ở trước)
        result = await self.deploy.ainvoke({
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
    FILE = os.getenv("FILE_NAME") # sẽ thay đổi sau này nhằm chạy được cho api
    CHROMA_HOST = os.getenv("CHROMA_CONTAINER_NAME")
    question = "Hãy tóm tắt file, sau đó tìm kiếm xem câu hỏi 5 trong file là hỏi về cái gì? hướng giải pháp của file là gì?. File có tổng cộng bao nhiêu câu hỏi cần giải quyết?" # user prompt

# Đây là ngữ cảnh của prompt (có thể tạo nhiều situation để chạy nhiều lần test AI)
    dap_an = asyncio.run(test.question(cau_hoi=question, file_path=FILE)) #Bắt đầu chạy AI theo lần lượt ngữ cảnh và câu hỏi
    print(dap_an)