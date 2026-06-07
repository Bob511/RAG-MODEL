import { hash } from "bcrypt";
import * as userServices from "../services/UserService.js";

export const getAllUsers = async (req, res) => {
  try {
    const users = await userServices.getAllUsers();
    return res.status(200).json(users);
  } catch (error) {
    return res.status(500).json({ error: "Lỗi kết nối với server!" });
  }
};

export const getAccount = async (req, res) => {
  try {
    const username = req.body.username;
    const password = req.body.password;
    const account = await userServices.getAccount(username, password);
    if (account === false) {
      return res.status(404).json({ status: "fail" });
    }
    userServices.setData("username", username);
    userServices.setData("is_admin", false);
    return res.status(200).json({ status: "success" });
  } catch (error) {
    console.error("Lỗi getAccount: ", error);
    res.status(500).json({ error: "Lỗi kết nối server!" });
  }
};

export const putUser = async (req, res) => {
  try {
    const username = req.body.username;
    const password = req.body.password;
    const email = req.body.email;
    const confirmPassword = req.body.confirmPassword;
    const success = await userServices.putUser(
      username,
      password,
      email,
      confirmPassword,
    );
    if (success === false) {
      return res
        .status(409)
        .json({ status: "fail", message: "Không tạo được account!" });
    }
    res
      .status(201)
      .json({ status: "success", message: "Khởi tạo thành công!" });
  } catch (error) {
    console.error("Lỗi: ", error);
    return res
      .status(500)
      .json({ status: "fail", message: "Lỗi kết nối server!" });
  }
};

export const deleteUser = async (req, res) => {
  try {
    const username = req.body.username;
    const success = await userServices.deleteUser(username);
    if (success === false) {
      return res
        .status(404)
        .json({ status: "fail", message: "User not found" });
    }
    return res.status(200).json({ status: "success", message: "User deleted" });
  } catch (error) {
    return res.status(500).json({ error: "Lỗi kết nối server!" });
  }
};
