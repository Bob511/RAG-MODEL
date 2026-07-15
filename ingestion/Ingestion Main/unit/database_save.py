import chromadb
import os
from dotenv import load_dotenv
load_dotenv()
import ollama

def push_to_chroma(final_chunks, file_path: str, author: str = "", website: str = ""):

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
    print("đã đưa tất cả lên database")
    # ... (Gọi collect.upsert như cũ) ...




def save_chunks_to_chromadb_cloud(chunks):
    ChromaAPI = os.getenv("ChromaAPI")
    """
    Nhúng các chunks bằng Ollama và đẩy thẳng lên ChromaDB Cloud.
    """
    print("☁️ Đang kết nối tới ChromaDB Cloud...")
    
    # 1. Khởi tạo CloudClient (Thay thế cho PersistentClient)
    client = chromadb.CloudClient(
    api_key= ChromaAPI,
    tenant='4798bb4f-8541-44e6-ab6f-6b6594fcef7a',
    database='BIZRAG'
    )
    COLLECTION_NAME = 'Bigchild'
    # 2. Tạo hoặc lấy Collection trên Cloud
    collection = client.get_or_create_collection(name = COLLECTION_NAME)
    
    documents = []
    metadatas = []
    ids = []
    embeddings = []

    print(f"🔄 Bắt đầu embedding và chuẩn bị dữ liệu cho {len(chunks)} chunks...")
    
    # 3. Duyệt qua từng chunk để lấy vector
    for i, chunk in enumerate(chunks):
        # Gọi Ollama chạy local để lấy vector
        response = ollama.embed(
            model='qwen3-embedding:0.6b', 
            input=chunk.page_content
        )
        vector = response['embeddings'][0]
        
        documents.append(chunk.page_content)
        
        meta = chunk.metadata if chunk.metadata else {}
        meta["chunk_id"] = i
        metadatas.append(meta)
        
        ids.append(f"chunk_{i}")
        embeddings.append(vector)
        
        print(f"  -> Đã nhúng xong chunk {i+1}/{len(chunks)}")

    # 4. Đẩy toàn bộ lên Cloud
    if documents:
        print("🚀 Đang tải dữ liệu lên ChromaDB Cloud... Vui lòng đợi.")
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print("\n✅ ĐÃ LƯU THÀNH CÔNG LÊN CHROMA CLOUD!")
    else:
        print("\n⚠️ Không có dữ liệu để lưu.")
        
    return collection