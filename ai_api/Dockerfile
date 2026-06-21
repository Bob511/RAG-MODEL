# Tạo python phiên bản cố định (mới và phù hơp cho các thư viện ML (vì mỗi phiên bản mới là mỗi lần chỉnh sửa code)
FROM python:3.10-slim 
# Tạo 1 file app và chuyển vào app đó (là sự kết hợp của cd /app và mkdir /app)
WORKDIR /app/ai_api
# sao chép file requirements.txt (chứa các thông tin về thư viện trong file AI)
COPY requirements.txt .
# tải các thư viện dựa trên requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# sao chép tất cả các folder và file ở cùng Dockerfile
COPY . .
# lệnh chạy ứng dụng
CMD ["python", "llm_client.py"]