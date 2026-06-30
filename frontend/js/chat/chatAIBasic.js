document.addEventListener("click", (event) => {
  if (flag == 1 && menu && !menu.contains(event.target)) {
    menu.style.display = "none";
    flag = 0;
  }
  if (
    !boxHistory.contains(event.target) &&
    !boxHistoryAfter.contains(event.target)
  ) {
    boxHistory.style.animation = "slideLeft 0.2s ease-in-out";
    setTimeout(() => {
      boxHistory.style.display = "none";
      boxHistoryAfter.style.display = "block";
    }, 100);
  }
});

let eventClick = 0;
// Sự kiện 2: Click vào nút thu nhỏ (☰) thì hiện lại menu chính
btnReopen.addEventListener("click", (event) => {
  event.stopPropagation(); // Ngăn sự kiện click lan ra ngoài gây đóng menu ngay lập tức
  boxHistory.style.animation = "slideRight 0.2s ease-in-out";
  setTimeout(() => {
    boxHistory.style.display = "block";
    boxHistoryAfter.style.display = "none";
  }, 100);
});

boxInput.addEventListener("input", function () {
  const defaultHeight = 60;
  const maxHeight = 47 * 3; // Giới hạn khoảng 3 dòng

  // 1. Reset chiều cao trước khi tính toán lại (Giúp ô input co lại được khi bạn xóa bớt chữ)
  this.style.height = `${defaultHeight}px`;

  // 2. Kiểm tra nếu trống thì giữ nguyên chiều cao mặc định và dừng lại
  if (input.value.trim() === "") return;

  // 3. Nếu nội dung dài hơn, tự động giãn ra nhưng không vượt quá maxHeight
  if (input.scrollHeight <= maxHeight) {
    this.style.height = `${input.scrollHeight}px`;
  } else {
    this.style.height = `${maxHeight}px`; // Cố định ở maxHeight nếu vượt quá
  }
});

input.addEventListener("keydown", function (event) {
  // Kiểm tra nếu phím nhấn là "Enter"
  if (event.key === "Enter" && !event.shiftKey) {
    // Ngăn chặn xuống dòng trong ô textarea nếu bạn không muốn nó tự nhảy dòng
    event.preventDefault();

    // Gọi hàm gửi
    send();
  }
});

// submit chat
const chatMain = document.querySelector(".chat-main");
const submit = document.querySelector(".btn-send");

function send() {
  const messageUser = document.getElementById("input").value.trim();
  if (messageUser !== "") {
    input.value = "";
    addMessage(messageUser, "user");
    boxInput.style.height = 56 + "px";
  }
}

// add to chat main user and bot
function addMessage(message, role) {
  const boxMessageAnonymous = document.createElement("div");
  const messageAnonymous = document.createElement("div");
  boxMessageAnonymous.classList.add("box-message", role);
  messageAnonymous.classList.add("message");
  messageAnonymous.innerHTML = message;
  boxMessageAnonymous.appendChild(messageAnonymous);
  chatMain.appendChild(boxMessageAnonymous);
}

//reset chat
function resetChat() {
  window.location.href = "/chat/chatAIver2";
}
