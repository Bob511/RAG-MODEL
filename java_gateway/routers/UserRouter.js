import * as userController from "../controllers/UserController.js";
import express from "express";
import { json } from "express";
import cors from "cors";

const router = express.Router();

router.use(json());
router.use(cors());

router.get("/", userController.getAllUsers);
router.post("/auth/register", userController.putUser);
router.post("/auth/login", userController.getAccount);
router.post("/auth/delete", userController.deleteUser);

export default router;
