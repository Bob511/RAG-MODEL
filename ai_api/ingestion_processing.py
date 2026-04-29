import json, chromadb, os

class read_and_write_chromaDB:
    def __init__(self, file = None, host_chroma = None):
        self.file = file # lấy tên file
        self.host = host_chroma # gán tên chroma ở phần container
        self.name = os.getenv("CHROMA_DATABASE")
    def _read_write_chroma(self):
        print("1. Reading....")
        try:
            # mở file chứa vector, chỉ đọc "r" với  utf-8 là chuẩn để dịch sang mọi ngôn ngữ, as file để viết tắt
            with open(f"{self.file}", 'r', encoding='utf-8') as file:
                # chạy và lưu file lại vào biến data_embed
                data_embed = json.load(file) # load dữ liệu file json ở trên lên thanh RAM (xử lý theo logic)
            print(f"Thành công lấy {len(data_embed)} dòng từ file")
        except FileNotFoundError: # nếu ko tìm thấy
            raise ValueError ("ERROR! ❌, Không tìm thấy file")
        print("2. Connecting to chromaDB")
        client = chromadb.HttpClient(host= f"{self.host}", port=8000) # lấy host là service name của docker-compose
        # HttpClient là hàm để thực hiện giao tiếp thông qua HTTP
        collection = client.get_or_create_collection(name=self.name) # tìm kiếm hoặc tạo ra 1 cơ sở dữ liệu (vector)
        print("3. Cutting...")
        docs, metadatas, ids = [], [], []
        for head in data_embed:
            every_id = head.get("id")

            # 1. Lấy cái hộp nhỏ "metadata"
            raw_metadata = head.get("metadata")

            # Kiểm tra an toàn: Đảm bảo raw_metadata là kiểu Dictionary
            if not isinstance(raw_metadata, dict): # kiểm tra key metada có phải thuộc dạng dict ko (chứa text, vector,..)
                raw_metadata = {} # nếu ko, cho thành rỗng, tránh lỗi

            # 2. Mở hộp nhỏ lấy "text"
            every_head = raw_metadata.get("text") # Tiếp tục tìm bên trong raw_metadata tìm xem có key text ko

            # 3. BỘ LỌC (Lọc bỏ nếu thiếu ID hoặc Text)
            if not every_head or not str(every_head).strip() or not every_id: # nếu không có text hoặc ko có ID  thì chạy tiếp
                continue

            # 4. TỐI ƯU DỮ LIỆU: Rút 'text' ra khỏi metadata để tránh lưu trùng lặp gây nặng DB
            clean_metadata = raw_metadata.copy() # Tạo bản sao để tránh thay đổi dữ liệu gốc
            if "text" in clean_metadata:
                del clean_metadata["text"] # Xóa key text đi

            # Nếu xóa xong mà metadata trống trơn, gán cho nó giá trị mặc định
            if not clean_metadata:
                clean_metadata = {"source": "unknown"}

            # 5. Lưu vào mảng
            docs.append(str(every_head))
            metadatas.append(clean_metadata)
            ids.append(str(every_id))
        print("4. Upload to ChromaDB")
        if len(docs) > 0:
            collection.add(
                documents= docs,
                metadatas= metadatas,
                ids= ids
            ) # Bỏ tài liệu vào cơ sở dữ liệu "tai_lieu_ai_2 (collection)"
        return collection
    def ask_ans(self, ques):
        collection = self._read_write_chroma()
        print("5. Testing")
        question = f"{ques}"
        res = collection.query(
        query_texts=[question],
        n_results= 5
        )   # tìm kiếm các thông tin có liên quan gần nhất với câu hỏi ques
        return res
        
if __name__ == "__main__":
    CHROMA_HOST = os.getenv("CHROMA_CONTAINER_NAME")
    FILE = os.getenv("FILE_NAME")
    test = read_and_write_chromaDB(host_chroma= CHROMA_HOST, file= FILE)
    test.ask_ans()