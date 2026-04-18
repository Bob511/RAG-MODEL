from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter
import ollama
import os
import fitz


def split_pdf_into_batches(input_pdf_path, output_folder="pdf_batches", pages_per_batch=5):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(input_pdf_path)
    total_pages = len(doc)
    batch_files = []
    
    for i in range(0, total_pages, pages_per_batch):
        new_doc = fitz.open()
        start_page = i
        end_page = min(i + pages_per_batch - 1, total_pages - 1)
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
        
        batch_filename = os.path.join(output_folder, f"batch_{start_page}_to_{end_page}.pdf")
        new_doc.save(batch_filename)
        new_doc.close()
        batch_files.append(batch_filename)
    doc.close()
    return batch_files

def process_pdf_with_docling(file_path):
    # 1. Sử dụng Docling để đọc PDF
    # Docling sẽ tự động nhận diện tiêu đề, bảng biểu và chuyển về Markdown chuẩn
    loader = DoclingLoader(file_path=file_path)
    
    # Load dữ liệu dưới dạng Document object của LangChain
    docs = loader.load()
    
    # 2. Chia nhỏ văn bản nhưng giữ cấu trúc Markdown
    # Vì Docling trả về Markdown, ta có thể cắt theo các đề mục #, ##
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]
    )
    print(text_splitter)
    # split_documents giúp giữ lại metadata (như tên file, số trang)
    splits = text_splitter.split_documents(docs)
    # 3. Chuyển đổi sang Vector bằng Qwen 0.6B trên Ollama
    vector_results = []
    print(f"Đang tạo embedding cho {len(splits)} đoạn văn bản...")
    
    for i, chunk in enumerate(splits):
        response = ollama.embed(
            model='qwen3-embedding:0.6b',
            input=chunk.page_content
        )
        vector_results.append({
            "id": i,
            "text": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": response['embeddings'][0]
        })
    
    return vector_results

# Thực thi
file_path = r"D:\RAG_MODEL\52500174(report_2)(2).pdf"
lst = split_pdf_into_batches(file_path, pages_per_batch=3)
for small in lst:
    converter = DocumentConverter()
    result = converter.convert(small)
    clean = process_pdf_with_docling(result)
    markdown_content = result.document.export_to_markdown(clean)
    print(markdown_content)
    print("Thành công!")