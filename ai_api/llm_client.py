from langchain_core.prompts import PromptTemplate # tạo langchain nhưng chỉ lấy core và phần prompts (để hạn chế ô nhớ và tối ưu tốc độ)
from langchain_ollama import OllamaLLM
import time, os
from langchain_core.chat_history import InMemoryChatMessageHistory
from check_ultis import check_database
from ingestion_processing import read_and_write_chromaDB
from dotenv import load_dotenv
load_dotenv()
# thử nghiệm xem chromaDB (hay database đã chạy chưa)
if not check_database():
    raise ValueError ("Database still unavailable, please try again!")

class BotAi:
    def __init__(self):
        # Mở đường hầm kết nối từ trong Docker ra phần mềm Ollama trên máy tính Windows
        HOST_OLLAMA = os.getenv("OLLAMA MODEL", "qwen2.5:3b")
        URL_OLLAMA = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        self.llm = OllamaLLM(
            model=HOST_OLLAMA,
            base_url=URL_OLLAMA
        )
        prompt = '''Bạn là trợ lý AI, chuyên phục vụ cho việc tìm hiểu, phân tích các thông tin từ file PDF (nếu có) và đưa ra câu trả lời 
        dựa vào những gì bạn đã được cung cấp (tin nhắn từ trước)
        Ngữ cảnh của file PDF: {ngu_canh}
        
        Câu trả lời trước đó: {lich_su}
        Câu hỏi của user: {cau_hoi}
        Trả lời: '''
        self.memory = InMemoryChatMessageHistory() # khởi tạo OOP (lưu trữ đoạn chat) vào trong biến memory
        self.template = PromptTemplate(
            input_variables= ["ngu_canh", "cau_hoi", "lich_su"],
            template=prompt
        )

        self.deploy = self.template | self.llm # tích hợp template vào chuỗi llms
    def question(self, ngu_canh, cau_hoi):
        print("Trả lời...")
        lich_su_hien_tai = "\n".join([f"{msg.type} : {msg.content}" for msg in self.memory.messages])
        # lần lượt tạo biến msg truy xuất trong memory.messages (nơi lưu trữ tất cả các tin nhắn)
        # msg.type() : dùng để hiển thị (human hoặc ai)
        # msg.content() : dùng để hiển thị nội dung 
        begin = time.time()
        #invoke để kích hoạt và chạy ai (deploy chạy ai nhờ vào | ở trước)
        result = self.deploy.invoke({
            "ngu_canh" : ngu_canh,
            "cau_hoi": cau_hoi,
            "lich_su": lich_su_hien_tai
            # hiển thị ra (vd: human: tôi là Dân; AI: chào Dân)
        })
        self.memory.add_user_message(cau_hoi) # gán câu hỏi của user
        self.memory.add_ai_message(result) # gán câu trả lời của ai vào self.memory (vốn là biến của OOP)
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
    dap_an = test.question(ngu_canh=ask, cau_hoi=question) #Bắt đầu chạy AI theo lần lượt ngữ cảnh và câu hỏi
    print(dap_an)