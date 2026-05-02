import os
from marker.models import load_all_models


def setup():
    """Khởi động Engine AI trên GPU và cấu hình thư mục lưu trữ."""
    os.environ["HF_HOME"] = "D:/Marker_Cache"
    return load_all_models()