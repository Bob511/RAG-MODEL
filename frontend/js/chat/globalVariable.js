require("dotenv").config();
// history
const boxHistory = document.querySelector(".box-menu");
const boxHistoryAfter = document.querySelector(".box-menu-after");
const btnReopen = document.querySelector(".menu-after");

// input
const boxInput = document.querySelector(".box-input");
const input = document.querySelector(".input");

// mode language
let turn = 0;
let modeLanguage = 0;
let modeModel = 0;

//nếu như flagModeLanguage === 1 thì là English/ === 2 thì là China
let flagModeLanguage = 0;
// nếu như flagModeModel === 1 thì là Thinking/ === 2 thì là Improve Prompt
let flagModeModel = 0;

const menu = document.querySelector(".menuChangeMode");
let flag = 0;
