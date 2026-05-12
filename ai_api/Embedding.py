import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
def get_embedding(text:str) -> list[float]:
    start = time.time()
    # gọi model và lưu trữ vào biến model_core
    model_core = GoogleGenerativeAIEmbeddings(model= "models/gemini-embedding-2")
    # Tiến hành covert text sang vector thông qua embed_query của langchain google 
    vector_convert = model_core.embed_query(text)
    end = time.time()
    time_run = end - start
    print(f"Time to run and convert text into vector is: {time_run:.4f} với kích thước vector là {len(vector_convert)} chiều")
    return vector_convert
