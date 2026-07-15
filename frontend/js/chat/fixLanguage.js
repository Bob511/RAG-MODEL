// submit chat bằng enter
// Thay vì dùng keyup, hãy dùng keydown để chặn hành vi nhảy dòng của Enter kịp thời
let isCheckingSpelling = false;
input.addEventListener("keydown", async (event) => {
  // LOGIC 1: Bấm Enter để gửi tin nhắn
  if (event.code === "Enter" && !event.shiftKey) {
    event.preventDefault(); // Chặn xuống dòng thành công 100% khi dùng keydown

    const messageUser = input.value.trim();

    if (messageUser !== "") {
      input.value = "";
      input.style.height = "48px"; // Reset chiều cao input
      addMessage(messageUser, "user");
    }
  }

  // LOGIC 2: Bấm Space để tự check và sửa chính tả từ vừa gõ
  if (turn === 1) {
    if (event.code === "Space") {
      if (isCheckingSpelling) return; // Nếu đang check rồi thì bỏ qua không gửi thêm

      const currentText = input.value; // Ghi nhớ đoạn text NGAY TẠI THỜI ĐIỂM bấm Space

      if (currentText.trim() !== "") {
        isCheckingSpelling = true;

        // 1. Gọi hàm check chính tả cho đoạn text tính đến lúc bấm Space
        let correctedText =
          flagModeLanguage === 1
            ? await checkFailLanguage(currentText, "en-US")
            : await checkFailLanguage(currentText, "zh-CN");

        // 2. Lấy nội dung MỚI NHẤT hiện tại trong ô input (lúc này có thể bạn đã gõ thêm chữ)
        const latestText = input.value;

        // 3. Lấy ra đoạn chữ bạn vừa gõ thêm trong lúc chờ API chạy ngầm
        // Bằng cách cắt chuỗi từ vị trí độ dài của currentText trở đi
        const textAfterSpace = latestText.substring(currentText.length);

        // 4. Cộng đoạn đã sửa sạch với đoạn gõ thêm phía sau + 1 dấu cách vừa bấm
        input.value = correctedText + textAfterSpace;

        isCheckingSpelling = false;

        // Kích hoạt lại sự kiện thay đổi chiều cao ô gõ nếu chữ dài ra
        input.dispatchEvent(new Event("input"));
      }
    }
  }
});

// Hàm xử lý dữ liệu API LanguageTool
// en-US - English
// zh-CN - China
async function checkFailLanguage(messageUser, language) {
  try {
    const response = await fetch("https://api.languagetool.org/v2/check", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        language: language,
        text: messageUser,
      }),
    });

    // Chuyển kết quả sang dạng Object JSON thay vì `.text()` thô
    const data = await response.json();
    console.log("Dữ liệu gốc từ API:", data);

    let textArray = messageUser.split("");

    // Duyệt ngược từ cuối danh sách lỗi lên đầu để không làm lệch chỉ số (index) vị trí của từ
    if (data.matches && data.matches.length > 0) {
      for (let i = data.matches.length - 1; i >= 0; i--) {
        const match = data.matches[i];

        // Nếu có từ gợi ý thay thế (replacements)
        if (match.replacements && match.replacements.length > 0) {
          const replacement = match.replacements[0].value; // Lấy từ gợi ý đầu tiên
          const offset = match.offset;
          const length = match.length;

          // Thay thế từ sai bằng từ đúng trong mảng ký tự
          textArray.splice(offset, length, replacement);
        }
      }
    }

    return textArray.join(""); // Trả về chuỗi chữ đã được sửa hoàn chỉnh
  } catch (error) {
    console.error("Lỗi API LanguageTool:", error);
    return messageUser; // Nếu lỗi API thì trả về lại đúng chữ user gõ ban đầu để không sập app
  }
}
