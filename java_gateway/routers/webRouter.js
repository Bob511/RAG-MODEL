import cors from "cors";
import express, { json } from "express";
import * as webController from "../controllers/webController.js";
const webRouter = express.Router();

webRouter.use(json());
webRouter.use(cors());

webRouter.get("/index", webController.getURL);
webRouter.get("/", webController.getURL);
webRouter.get("/user/infoUser", webController.getURL);
webRouter.get("/login", webController.getURLAI);
webRouter.get("/register", webController.getURLAI);
webRouter.get("/logout", webController.getURL);

webRouter.get("/chat/chatAI", webController.getURLAI);
webRouter.get("/function/article", webController.getURLAI);
webRouter.get("/fail/404", webController.getURLAI);

// else

export default webRouter;
