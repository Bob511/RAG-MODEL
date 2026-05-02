import chromadb
import os
from dotenv import load_dotenv
load_dotenv()
import ollama

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