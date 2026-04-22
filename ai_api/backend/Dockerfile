# 1. Chọn hệ điều hành nền có sẵn Python
FROM python:3.10-slim
# 2. Thiết lập thư mục làm việc bên trong container
WORKDIR /app
# 3. Copy file danh sách thư viện vào trước
COPY requirements.txt .
# 4. Cài đặt các thư viện (Docker sẽ nhớ bước này, không phải cài lại mỗi lần)
RUN pip install --no-cache-dir -r requirements.txt
# 5. Copy toàn bộ code từ máy mình vào trong container
COPY . .
# 6. Mở cổng 8000 để bên ngoài có thể truy cập
EXPOSE 8000
# 7. Lệnh để chạy server khi container khởi động
CMD ["uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8000"]