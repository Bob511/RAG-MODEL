# Mục tiêu file dùng để test và kiểm tra xem database bên chroma đã hoạt động và kết nối được chưa.
import time
import requests
# URL: chroma_system là tên của container ở phần docker-compose.yml của Dan
# :8000 là port số 8000 được định nghĩa trong docker-compose.yml
def check_database(url = "http://chroma_system:8000", retries = 10, delay = 5):
    print("Testing...")
    for i in range(retries):
        try:
            # get là hàm của thư viện requests để lấy thông tin từ web có link là:
            # api/v2/heartbeat, cấu trúc này là quy định của chromaDB nên không thể chỉnh
            respone = requests.get(f"{url}/api/v2/heartbeat")
            # 200 là tình trạng thành công của việc chạy code (lấy được infor của link)
            if respone.status_code == 200:
                print("✅ Kết nối thành công! Database đã sẵn sàng.")
                return True
            else:
                print("😪, Please wait for a while")
                time.sleep(delay)
        # nếu không lấy dược, thử lại    
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Database chưa sẵn sàng... Thử lại lần {i+1}/{retries} (chờ {delay}s)")
            time.sleep(delay)
    print(f"Error!!❌, lỗi dính phải là: {respone.status_code} và {respone.text}")
    return False

if __name__ == "__main__":
    check_database()