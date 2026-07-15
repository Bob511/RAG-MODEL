document.body.addEventListener("keydown", (event) => {
  if (event.key === "Tab") {
    event.preventDefault();
    input.focus();
  }
});
