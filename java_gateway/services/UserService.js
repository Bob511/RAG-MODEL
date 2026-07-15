import { format } from "mysql2";
import connection from "../config/db.js";
import bcrypt, { hash } from "bcrypt";

//lưu username , is_admin vào ram
const dataUser = {
  username: null,
  is_admin: false,
};

async function getData(gen) {
  if (gen == "username") {
    return dataUser.username;
  } else if (gen == false) {
    return dataUser.is_admin;
  }
}

async function setData(gen, data) {
  if (gen == "username") {
    dataUser.username = data;
  } else if (gen == false) {
    dataUser.is_admin = data;
  }
}

async function getAllUsers() {
  try {
    const [rows] = await connection.execute("SELECT * FROM accounts");
    return rows;
  } catch (error) {
    console.error("Lỗi kết nối với server!");
    return false;
  }
}

async function getUser(username) {
  try {
    const [rows] = await connection.execute(
      "SELECT * FROM accounts WHERE username = ?",
      [username],
    );
    return rows[0] || false;
  } catch (error) {
    console.error("Lỗi getUser:", error);
    return false;
  }
}

async function isPassword(password, hashPassword) {
  try {
    const isCheck = await bcrypt.compare(password, hashPassword);
    return isCheck;
  } catch (error) {
    console.error("Sai password");
    return false;
  }
}

async function getAccount(username, password) {
  try {
    const isCheckUsername = await getUser(username);
    if (isCheckUsername === false) {
      return false;
    }
    const [rows] = await connection.execute(
      "SELECT * FROM accounts WHERE username = ?",
      [username],
    );
    const isCheckPass = await isPassword(password, rows[0].password);
    if (isCheckPass === false) {
      return false;
    }
    return true;
  } catch (error) {
    console.error("Lỗi getAccount: ", error);
    return false;
  }
}

async function getEmail(userEmail) {
  try {
    const [rows] = await connection.execute(
      "SELECT email FROM accounts where email = ?",
      [userEmail],
    );
    return rows[0] || false;
  } catch (error) {
    return false;
  }
}

async function putUser(username, password, email, confirmPassword) {
  try {
    //check xem hai password trùng nhau không
    const isComparePass = password === confirmPassword;
    if (isComparePass === false) {
      return false;
    }

    //check trùng mail
    const isEmail = await getEmail(email);
    if (isEmail === true) {
      return false;
    }

    //check trùng username
    const isUsername = await getAccount(username, password);
    if (isUsername === true) {
      return false;
    }

    //check
    const hash = await bcrypt.hash(password, 10);
    const [result] = await connection.execute(
      "INSERT INTO accounts (username, password, email) VALUES (?,?,?)",
      [username, hash, email],
    );

    return true;
  } catch (error) {
    console.error("Lỗi putUser:", error);
    return false;
  }
}

async function deleteUser(username) {
  try {
    const isExisted = await getUser(username);
    if (!isExisted) {
      console.error("Không có user!");
      return false;
    }
    const [result] = await connection.execute(
      "DELETE FROM accounts WHERE username = ?",
      [username],
    );
    return result.affectedRows > 0;
  } catch (error) {
    console.error("Lỗi delete user : ", error);
    return false;
  }
}

export { getData, setData, getAllUsers, getAccount, putUser, deleteUser };
