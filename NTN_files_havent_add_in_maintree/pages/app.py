import streamlit as st
import json

def load_user_data():
    try:
        with open("user_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    
def save_user_data(user_data):
    try:
        data = load_user_data()
        data.append(user_data)
        with open("user_data.json", "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)    
    except FileNotFoundError:
            return []

if  "username" not in st.session_state:
    st.session_state['username'] = "khách"
if  "password" not in st.session_state:
    st.session_state['password'] = ""

def trang_chu():
    st.subheader("Trang chủ")
    if(st.session_state['username'] != "khách"):
        st.write(f"Chào mừng {st.session_state['username']} đến với trang chủ của dự án lớn!")
    st.write("Chào mừng bạn đến với trang chủ của dự án lớn!")
 
def ai():
    st.subheader("AI")
    st.write("Đây là phần AI của dự án lớn!")

def check_username(username_new):
    data = load_user_data()
    for user in data:
        if(user["username"] == username_new):
            return True
        else:
            return False

def check_password(password_new):
    data = load_user_data()  
    for password in data:
        if(password["password"] == password_new):
            return True
        else:
            return False
        
def dang_nhap():
    st.subheader("Đăng nhập")
    with st.form("Đăng nhập"):
        username = st.text_input("Nhập tên đăng nhập", placeholder="Nhập username vào...")
        password = st.text_input("Nhập mật khẩu", placeholder="Nhập password vào...", type="password")
        submitted = st.form_submit_button("Đăng nhập")
        if submitted:
            if(check_username(username)):
                if(check_password(password)):
                    st.toast("Đang nhập thành công!")
                    st.session_state['username'] = username
                    st.session_state['password'] = password
            else:
                st.write("Sai tài khoản hoặc mật khẩu!")        
def dang_ki():
    with st.form("Đăng kí"):
        username = st.text_input("Nhập tên đăng nhập", placeholder="Nhập username vào...")
        password = st.text_input("Nhập mật khẩu", placeholder="Nhập password vào...", type="password")
        password_verify = st.text_input("Nhập mật khẩu", placeholder="Nhập lại password vào...", type="password")
        submitted = st.form_submit_button("Đăng kí")
        if submitted:
            if(check_username(username)):
                st.toast("Tài khoản này đã tồn tại!")
            else:
                if(password == password_verify):
                    st.session_state['username'] = username
                    st.session_state['password'] = password
                    data_new = {"username": st.session_state['username'], "password": st.session_state['password']}
                    save_user_data(data_new)
                else:
                    st.toast("Mật khẩu không khớp!")

def info():
    st.subheader("Thông tin cá nhân")
    with st.form("Thông tin cá nhân"):
        username = st.text_input("Tên đăng nhập", value=st.session_state['username'], disabled=True)
        password = st.text_input("Mật khẩu", value=st.session_state['password'], disabled=True, type="password")
        submitted = st.form_submit_button("Cập nhật thông tin")
        dangxuat = st.form_submit_button("Đăng xuất")
        if submitted:
            st.write("Thông tin đã được cập nhật!")
        if(dangxuat):
            dang_xuat()

            
def dang_xuat():
    st.session_state.clear()
    st.toast("Đăng xuất...")
    st.switch_page("app.py")

st.set_page_config("Du an lon", "📊")

with st.sidebar:
    st.header("Menu")

choice = st.sidebar.radio(
    "-------------------",
    ["Trang chủ", "AI", "Thông tin cá nhân" if st.session_state['username'] != "khách" else "Đăng nhập", "Đăng kí" if st.session_state['username'] == "khách" else ""]
)
if(choice == "Trang chủ"):
    trang_chu()
elif(choice == "AI"):
    ai()
elif(choice == "Đăng nhập"):
    dang_nhap()
elif(choice == "Đăng kí"):
    dang_ki()
elif(choice == "Thông tin cá nhân"):
    info()
