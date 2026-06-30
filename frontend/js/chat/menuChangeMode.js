// change modeLanguage turn 0 tắt, turn 1 bật
// case 1: english modeLanguage = 1;
// case 2: china modeLanguage = 2;

// change modeModel
// case 1: thinking modeModel = 1;
// case 2: improve prompt  modeModel = 2;
// bật menu mode
function menuChangeMode(event) {
  // Ngăn chặn sự kiện click lan ra ngoài làm đóng menu ngay lập tức
  if (event) event.stopPropagation();

  if (flag === 0) {
    // Mở menu
    menu.style.display = "flex"; // Chuyển sang flex để khớp với CSS của bạn

    // Dùng setTimeout nhỏ để trình duyệt kịp cập nhật display trước khi áp dụng transform
    requestAnimationFrame(() => {
      menu.style.opacity = "1";
      menu.style.transform = "translateY(0)";
    });
    menu.style.animation = "slideUp 0.3s ease forwards";

    flag = 1;
  } else {
    // Đóng menu
    menu.style.opacity = "0";
    menu.style.transform = "translateY(10px)";

    menu.style.animation = "slideDown 0.3s ease forwards";

    // Đợi hiệu ứng chuyển động xong (ví dụ 300ms) rồi mới ẩn hẳn
    setTimeout(() => {
      if (flag === 0) {
        // Kiểm tra lại flag để tránh trường hợp người dùng click mở lại thật nhanh
        menu.style.display = "none";
      }
    }, 300);

    flag = 0;
  }
}

function modeL(mode) {
  switch (mode) {
    case 1:
      if (flagModeLanguage === 1) {
        flagModeLanguage -= 1;
      } else if (flagModeLanguage === 2) {
        flagModeLanguage -= 1;
      } else {
        flagModeLanguage += 1;
      }
      break;
    case 2:
      if (flagModeLanguage === 2) {
        flagModeLanguage -= 2;
      } else if (flagModeLanguage === 1) {
        flagModeLanguage += 1;
      } else {
        flagModeLanguage += 2;
      }
      break;
  }
  if (flagModeLanguage === 1 || flagModeLanguage === 2) {
    turn = 1;
  } else {
    turn = 0;
  }
  updateUI();
}

function modeM(mode) {
  switch (mode) {
    case 1:
      if (flagModeModel === 1 || flagModeModel === 3) {
        flagModeModel -= 1;
      } else {
        flagModeModel += 1;
      }
      break;
    case 2:
      if (flagModeModel === 2 || flagModeModel === 3) {
        flagModeModel -= 2;
      } else {
        flagModeModel += 2;
        improvePrompt();
        setTimeout(() => {
          flagModeModel -= 2;
          updateUI();
        }, 500);
      }
      break;
  }
  updateUI();
}

function updateUI() {
  const fixEnglish = document.querySelector("#fixEnglish");
  const fixChina = document.querySelector("#fixChina");

  const thinkingBtn = document.querySelector("#thinking");
  const improvePromptBtn = document.querySelector("#improve-prompt");

  if (fixEnglish) {
    fixEnglish.style.color = "";
    fixEnglish.style.border = "";
  }
  if (fixChina) {
    fixChina.style.color = "";
    fixChina.style.border = "";
  }

  // Áp dụng màu theo flag hiện tại
  if (flagModeLanguage === 1 && fixEnglish) {
    fixEnglish.style.color = "rgb(144, 32, 229)";
    fixEnglish.style.border = "1px solid rgb(144, 32, 229)";
  }

  if (flagModeLanguage === 2 && fixChina) {
    fixChina.style.color = "rgb(144, 32, 229)";
    fixChina.style.border = "1px solid rgb(144, 32, 229)";
  }

  if (flagModeLanguage === 3) {
    if (fixEnglish) {
      fixEnglish.style.color = "rgb(144, 32, 229)";
      fixEnglish.style.border = "1px solid rgb(144, 32, 229)";
    }
    if (fixChina) {
      fixChina.style.color = "rgb(144, 32, 229)";
      fixChina.style.border = "1px solid rgb(144, 32, 229)";
    }
  }

  if (thinkingBtn) {
    thinkingBtn.style.color = "";
    thinkingBtn.style.border = "";
  }
  if (improvePromptBtn) {
    improvePromptBtn.style.color = "";
    improvePromptBtn.style.border = "";
  }

  if (flagModeModel === 1 && thinkingBtn) {
    thinkingBtn.style.color = "rgb(144, 32, 229)";
    thinkingBtn.style.border = "1px solid rgb(144, 32, 229)";
  }

  if (flagModeModel === 2 && improvePromptBtn) {
    improvePromptBtn.style.color = "rgb(144, 32, 229)";
    improvePromptBtn.style.border = "1px solid rgb(144, 32, 229)";
  }

  if (flagModeModel === 3) {
    if (thinkingBtn) {
      thinkingBtn.style.color = "rgb(144, 32, 229)";
      thinkingBtn.style.border = "1px solid rgb(144, 32, 229)";
    }
    if (improvePromptBtn) {
      improvePromptBtn.style.color = "rgb(144, 32, 229)";
      improvePromptBtn.style.border = "1px solid rgb(144, 32, 229)";
    }
  }
}
