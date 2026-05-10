import path from "path";
import fs from "fs/promises";

const __dirname = import.meta.dirname;

async function handleURL(name) {
  try {
    if (name == "/") {
      name = "index";
    }
    name += ".html";

    const filePaths = {
      header: path.join(__dirname, "../../frontend/header/header.html"),
      footer: path.join(__dirname, "../../frontend/footer/footer.html"),
      main: path.join(__dirname, "../../frontend/", `views/${name}`),
    };

    const [headerHTML, mainHTML, footerHTML] = await Promise.all([
      fs.readFile(filePaths.header, "utf-8"),
      fs.readFile(filePaths.main, "utf-8"),
      fs.readFile(filePaths.footer, "utf-8"),
    ]);

    return headerHTML + mainHTML + footerHTML;
  } catch (error) {
    console.error("Lỗi đọc file:", error);
    return false;
  }
}

async function handleURLAI(name) {
  try {
    name += ".html";

    const filePaths = {
      main: path.join(__dirname, "../../frontend/", `views/${name}`),
    };

    const mainHTML = await fs.readFile(filePaths.main, "utf-8");

    return mainHTML;
  } catch (error) {
    console.error("Lỗi đọc file:", error);
    return false;
  }
}

export { handleURL, handleURLAI };
