async function AI(message) {
  const apiKey = process.env.API_KEY;

  const inputTarget = document.getElementById("input");
  const temp = inputTarget.value.trim();

  if (inputTarget) {
    inputTarget.value = ""; // Xóa sạch dữ liệu cũ trong ô input trước khi bắt đầu stream
  }

  try {
    // Dùng trực tiếp URL chuẩn của OpenRouter
    const apiUrl = "https://openrouter.ai/api/v1/chat/completions";

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: AI,
        stream: true,
        messages: [
          {
            role: "user",
            content: message,
          },
        ],
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // --- HÀNH ĐỘNG TRƯỚC KHI STREAM ---
    // Hiển thị khung gợi ý
    box_suggest.style.display = "flex";
    suggest.innerHTML = `<span class="thinking-text">Đang suy nghĩ...</span>`;

    // Đổ dữ liệu mẫu (example) vào trước nếu có
    if (Array.isArray(example)) {
      example.forEach((item) => {
        suggest.innerHTML += item;
      });
    }

    const reader = response.body
      .pipeThrough(new TextDecoderStream())
      .getReader();

    let buffer = "";
    let isFirstToken = true; // Cờ đánh dấu để xóa dữ liệu mẫu khi chữ từ AI bắt đầu đổ về

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += value;
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Giữ lại dòng cuối chưa hoàn chỉnh

      for (const line of lines) {
        const cleanedLine = line.trim();

        if (!cleanedLine || cleanedLine === "data: [DONE]") continue;

        if (cleanedLine.startsWith("data: ")) {
          try {
            const parsed = JSON.parse(cleanedLine.slice(6));
            const token = parsed.choices[0]?.delta?.content;

            if (token && inputTarget) {
              // Nếu nhận được chữ đầu tiên từ AI, xóa đống dữ liệu mẫu/ví dụ đi
              if (isFirstToken) {
                suggest.innerHTML = "";
                isFirstToken = false; // Đổi cờ để các chữ sau không bị xóa nữa
              }

              // Cộng dồn từng chữ (token) vào ô hiển thị
              suggest.innerHTML += token;

              // Tự động cuộn ô suggest xuống đáy khi chữ dài ra
              suggest.scrollTop = suggest.scrollHeight;
            }
          } catch (e) {
            // Bỏ qua lỗi nếu dòng dữ liệu JSON bị cắt đôi nửa chừng do mạng lag
          }
        }
      }
    }
  } catch (error) {
    console.log("Lỗi khi kết nối OpenRouter:", error);
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
