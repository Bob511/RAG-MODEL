'''LOGIC: Mục đích chính là chạy local up lên database bao gồm file nhúng và file chunking. Thiết kế database đã được cung cấp từ trước.
STEP: CHUNKING -> chạy file ingestion.py -> bỏ vào database với yêu cầu (ID, chuking, vector hóa) và table thông tin (Tên file gốc, tên tác giả/ web (nếu có))
LƯU Ý: Đây sẽ là file chạy local và upload lên database global, yêu cầu cho file input sẽ là (file chunking gồm metadata đi kèm. file vector hóa)'''
import json, chromadb, os, time
from dotenv import load_dotenv
load_dotenv()
class upload_database:
    def __init__(self, file = None):
        self.file = file # lấy tên file
        self.name = os.getenv("CHROMA_DATABASE")
    def _write_chroma(self):
        docs, metadata, ids = [], [], []
        chunk_count = 1
   
        print("1. LOADING...")
        try:
            # mở file chứa vector, chỉ đọc "r" với  utf-8 là chuẩn để dịch sang mọi ngôn ngữ, as file để viết tắt
            with open(f"{self.file}", 'r', encoding='utf-8') as file:
                # chạy và lưu file lại vào biến data_embed
                current_chunk = []
                if type(file) != dict:
                    for line in file:
                        if line.strip() == "--Chunk--":
                            text_content = "\n".join(current_chunk).strip()
                            if text_content:
                                docs.append(text_content)
                                metadata.append({"source": self.file, "index" : chunk_count})
                                ids.append(f"quiz_chunk_{chunk_count}")
                                chunk_count += 1
                                current_chunk = []
                        else:
                            current_chunk.append(line)
                else:
                    data_embed = json.load(file) # load dữ liệu file json ở trên lên thanh RAM (xử lý theo logic)
                    print(f"Successfully read {len(data_embed)} chunk from file")
                    for head in data_embed:
                        head: dict
                        # 1. Lấy cái hộp nhỏ "metadata"
                        raw_metadata = head.get("metadatas") # thông tin metadatas. Dictionary type     
                        # Kiểm tra an toàn: Đảm bảo raw_metadata là kiểu Dictionary
                        if not isinstance(raw_metadata, dict): # kiểm tra key metada có phải thuộc dạng dict ko (chứa text, vector,..)
                            raw_metadata = {} # nếu ko, cho thành rỗng, tránh lỗi
                        every_head = raw_metadata.get("document") # Lấy content 
                        every_id = raw_metadata.get("ids") # lấy id chunk
                        print(every_head, "\n\n\n")
                        print(type(every_head))
                        print(every_id)
                        # 3. BỘ LỌC (Lọc bỏ nếu thiếu ID hoặc Text)
                        if not every_head or not str(every_head).strip() or not every_id: # nếu không có text hoặc ko có ID  thì chạy tiếp
                            continue
                        # 4. TỐI ƯU DỮ LIỆU: Rút 'text' ra khỏi metadata để tránh lưu trùng lặp gây nặng DB
                        clean_metadata = raw_metadata.copy() # Tạo bản sao để tránh thay đổi dữ liệu gốc
                        if "document" in clean_metadata:
                            del clean_metadata["document"] # Xóa key text đi
                        # Nếu xóa xong mà metadata trống trơn, gán cho nó giá trị mặc định
                        if not clean_metadata:
                            clean_metadata = {"source": "unknown"}
                        # 5. Lưu vào mảng
                        docs.append(str(every_head))
                        metadata.append(clean_metadata)
                        ids.append(str(every_id))
        except FileNotFoundError: # nếu ko tìm thấy
            raise ValueError ("ERROR! ❌, Không tìm thấy file")
        print("2. CONNECTING TO CHROMADB")
        chroma_client = chromadb.CloudClient(
            database= "VectorManagement", 
            tenant= os.getenv("TENANT"),
            api_key= os.getenv("CHROMA_API"))
        collection = chroma_client.get_or_create_collection(name="VectorChunk") # tạo hoặc mở table với name ContentManagement
        
        # Thắc mắc: có thể upload lên chroma database thông qua list? hay cần vòng lặp?
        print("3. Upload to ChromaDB")
        if len(docs) > 0:
            collection.add(
                documents= docs,
                metadatas= metadata,
                ids= ids
            ) 
        print("UPDLOAD thành công")
        
if __name__ == "__main__":
    '''future: cần 1 chức năng auto lấy output file làm input bên đây để chạy local và up lên database'''
    FILE = os.getenv("FILE_NAME")
    print(FILE)
    test = upload_database(file= FILE)
    test._write_chroma()