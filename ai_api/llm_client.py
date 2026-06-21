''' MỤC TIÊU CHÍNH: khởi tạo LLMs và tìm kiếm các thông tin trên database bằng hybrid search. 
Với input: user prompt, file pdf của user (nếu có). 
Với output: JSON với nội dung chính là câu trả lời của LLMs
DATAFLOW: FastAPI -> LLm_client.py -> nhúng câu hỏi user -> tìm thông tin liên quan trong database với hybrid search (BM25 - Ensemble) 
-> chọn lọc các thông tin chunk có liên quan nhất bằng Jina -> Trả ra kết quả database -> đưa vào LLMs với prompt input(user pronpt + thông tin liên quan)
-> Trả ra kết quả cuối cùng là dạng JSON'''
from langchain_core.prompts import PromptTemplate # tạo langchain nhưng chỉ lấy core và phần prompts (để hạn chế ô nhớ và tối ưu tốc độ)
import time, os, asyncio, chromadb
from langchain_core.documents import Document
from typing import Generator, AsyncGenerator
# 1. Cập nhật cách gọi Chroma theo tiêu chuẩn gói độc lập
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import JinaRerank
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

class BotAi:
    def __init__(self):
       # callAPI từ groq với LOGIC: cần apikey, cần tên model, cần url (nếu không sử dụng thư viện), thêm nhiệt độ (nếu cần)
       # Cần 1 promptTemplate để làm khuôn (LLMs hiểu dễ hơn)
        self.llm = ChatGroq(model=os.getenv("MODEL"), temperature=0.7, api_key= os.getenv("API_KEY_GROQ"))

        prompt = '''Bạn là trợ lý AI, chuyên phục vụ cho việc tìm hiểu, phân tích các thông tin từ file PDF (nếu có) và đưa ra câu trả lời 
        dựa vào những gì bạn đã được cung cấp. Yêu cầu đưa ra ngôn ngữ trả lời phụ thuộc vào câu hỏi của user và ghi chú nguồn và số trang cụ thể (nếu là PDF)
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
        self.access_cloud = chromadb.CloudClient(tenant=os.getenv("TENANT"), api_key= os.getenv("CHROMA_API"), database= "VectorManagement")
    def stream_text(self) -> Generator[Document, None, None]:
        # Mục đích: cấp data cho BM25
        access_database = self.access_cloud.get_collection(name="VectorChunk")
        get_docs = access_database.get(include=['documents', 'metadatas'])
        doc_list = []
        for doc_text, meta in zip(get_docs['documents'], get_docs['metadatas']):
            doc_list.append(Document(page_content=doc_text, metadata= meta))
        return doc_list # trả về toàn bộ file database.
    
    def hybrid_search(self):
        # Mục Đích: Kích hoạt và sử dụng 2 mô hình tìm kiếm (BM25 + Chroma) và bộ lọc Jina Rerank
        doc = list(self.stream_text())
        bm25_retrievers = BM25Retriever.from_documents(doc)
        bm25_retrievers.k = 10
        chroma_vectorstore = Chroma(
            client= self.access_cloud,
            collection_name="VectorManagement",
            embedding_function= None
        )
        chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": bm25_retrievers.k})
        # 3. Hợp nhất hai bộ truy xuất chạy song song bằng EnsembleRetriever
        print("dùng Hybrid Search: \n")
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retrievers, chroma_retriever],
            weights=[0.5, 0.5] # Phân bổ trọng số cân bằng 50% từ khóa - 50% ngữ nghĩa
        )
        jina_compress = JinaRerank(jina_api_key=os.getenv("JINA_API"), top_n=5)
        self.hybrid_retrievers = ContextualCompressionRetriever(base_compressor=jina_compress, base_retriever=ensemble)
    
    async def question(self, cau_hoi : str) -> AsyncGenerator[str, None]:
        # Gọi LLMs phân rã thành 3 câu nhỏ + query DBS + trả về thông tin context + LLMs trả lời 
        print("Trả lời...")
        begin = time.time()
        if not self.hybrid_retrievers:
            self.hybrid_search()
        sub_questions = await self.decompose.ainvoke({"cau_hoi": cau_hoi})
        # Cắt chuỗi thành mảng các câu hỏi phụ - prompt có nếu ra - và \n -> thay thế - và dùng \n tách thành 1 đoạn
        sub_ques_query = [q.replace("-", "").strip() for q in sub_questions.content.split('\n') if q.strip()]
        print([doc for doc in sub_ques_query])
        # chạy for để tìm kiếm thông tin trên database thông qua hybrid search (BM25 + Chroma)
        store_relevant_docs = []
        for i in sub_ques_query:
            relevant_docs = await self.hybrid_retrievers.base_retriever.ainvoke(i) 
            # sử dụng await cho ainvoke() - Document type() vì đây là search từ database nên cần document lưu trữ docs và metadata
            # relevant_docs: Document typle, content = "nội dung trong database với k = 10"
            store_relevant_docs.extend(relevant_docs) # ko còn là list(list(document)) mà chỉ là list(document)
            # extend() dùng để thêm từng phần tử trong list, tuple vào list. Tức là thay vì thêm 1 mục (list hoặc tuple) thì đây sẽ là thêm từng phần tử
        unique_docs = []
        seen = set()
        # lọc chunk trùng (Lỗi hiện tại: document type nên không thể append() )
        for i in store_relevant_docs:
            i : Document # Vì IDE của VS ko thể biết i là kiểu biến gì trước khi chạy, do đó cần phân loại cho nó để page_content có thể đề xuất
            text = i.page_content
            if text not in seen: # dùng text vì set() sẽ chỉ băm các type iteration. Documents (i) là non-iteration nên phải dùng text 
                seen.add(text)
                unique_docs.append(i)
        # đánh giá bằng jina rerank. Trả về list(document)
        final_docs = self.hybrid_retrievers.base_compressor.compress_documents(documents=unique_docs, query=cau_hoi)
        # Việc dùng cloudclient cho vector -> ko up lên RAM thay vào đó tìm trên databse -> jina tính chính xác hơn, tuy nhiên vẫn chưa hoàn hảo
        context_part = []
        # trích xuất page_content và metadata (index và source)
        for doc in final_docs:
            doc : Document
            source = doc.metadata.get("source", "unkown")
            index = doc.metadata.get("index", "unknown")
            content = doc.page_content
            context_part.append(
                f"source: {source} | index: {index} \n {content}"
            )
        context = "\n".join(context_part)
        # ngữ cảnh
        stop = time.time()
        track_time = stop - begin
        print("Thời gian tìm kiếm và trả kết quả: ", track_time)
        begin = time.time()
        stop = time.time()
        track_time = stop - begin
        #invoke để kích hoạt và chạy ai (deploy chạy ai nhờ vào | ở trước)
        print(f"----- time to run AI is: {track_time:.4f}s -----")
        async for token in self.deploy.astream({ # sẽ tìm hiểu kỹ hơn hàm này trong tương lai
            "ngu_canh" : context,
            "cau_hoi": cau_hoi
        }):
            yield token.content # sẽ tìm hiểu kỹ khác biệt yield và return
        
    async def run(self):
        question = "Hãy tóm tắt file đại số tuyến tính, sau đó tìm kiếm xem câu hỏi 6 trong file là hỏi về cái gì? hướng giải pháp của file là gì?. File có tổng cộng bao nhiêu câu hỏi cần giải quyết?, theo bạn thì câu nào sẽ là khó nhất nhưng cơ sở nhất cho sau này?"
        async for chunk in self.question(question):
            print(chunk, end="", flush= True) # sẽ tìm hiểu tại sao lại cần dùng hàm này
        
if __name__ == '__main__':
    test = BotAi()
    asyncio.run(test.run())

# Điểm thiếu: chưa làm xong phần tích họp asyncio.gather (vì list lồng list lồng Document). Sẽ chỉnh sửa phần question sao cho nó là "động"