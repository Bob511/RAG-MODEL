import { handleURL, handleURLAI } from "../services/webService.js";

export const getURL = async (req, res) => {
  try {
    const linkFossil = req.path;
    const isCheck = await handleURL(linkFossil);
    if (isCheck === false) {
      return res.status(404).json({ status: "fail" });
    }
    res.send(isCheck);
  } catch (error) {
    console.error("Lỗi: ", error);
  }
};

export const getURLAI = async (req, res) => {
  try {
    const linkFossil = req.path;
    const isCheck = await handleURLAI(linkFossil);
    if (isCheck === false) {
      return res.status(404).json({ status: "fail" });
    }
    res.send(isCheck);
  } catch (error) {
    console.error("Lỗi: ", error);
  }
};
