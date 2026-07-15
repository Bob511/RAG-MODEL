import express, { json } from "express";
import cors from "cors";
import userRouter from "./routers/UserRouter.js";
import webRouter from "./routers/webRouter.js";
import db from "./config/db.js";

import session from "express-session";

import path from "path";
import { fileURLToPath } from "url"; // Dòng này cực kỳ quan trọng để sửa lỗi của bạn

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

app.use(json());
app.use(cors());

//dùng router cú pháp es6+, không dùng require
app.use("/users", userRouter);
app.use("/", webRouter);
// Trong file server.js hoặc app.js
app.use(express.static(path.join(__dirname, "../frontend")));
const PORT = 5001;
app.listen(PORT, () => {
  console.log(`Server chạy cổng ${PORT}`);
});
