let suggest = document.querySelector("#promptText");
let box_suggest = document.querySelector(".box-suggest");
const example = "Đang Suy Nghĩ...";

async function improvePrompt(mode) {
  let AI = "";
  let prompt = "";
  if (mode === 1) {
    AI = "nvidia/nemotron-3-ultra-550b-a55b:free";
    prompt = `Bạn là một chuyên gia phân tích tâm lý và chiến lược. Hãy đọc kỹ nội dung dưới đây và phân tích chi tiết theo các mục sau:

Tóm tắt cốt lõi: Nội dung này thực chất đang nói về điều gì?

Ý định của người gửi: Yêu cầu bề nổi và động cơ ngầm (nếu có) của họ là gì?

Các điểm cốt yếu cần lưu ý: Những chi tiết quan trọng, rủi ro hoặc cơ hội cần chú ý.

Góc nhìn & Đề xuất: Suy nghĩ khách quan của bạn và hướng xử lý tiếp theo cho tôi.`;
  } else if (mode === 2) {
    AI = "poolside/laguna-xs.2:free";
    prompt = `Viết lại câu lệnh dưới đây thành một prompt chi tiết, đầy đủ và chuyên nghiệp hơn bằng tiếng Việt.YÊU CẦU: Chỉ trả về câu prompt mới trong khối , không giải thích, không chào hỏi.Câu lệnh cần viết lại`;
  }
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
            content: prompt + temp,
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

function btnDelete() {
  suggest.innerHTML = "";
  box_suggest.style.display = "none";
}

function copyText() {
  try {
    navigator.clipboard.writeText(suggest.innerHTML.trim());
    alert("Đã sao chép thành công");
  } catch (error) {
    alert("Sao chép không thành công");
  }
}

function selectPrompt() {
  input.value = suggest.innerHTML.trim();
}
