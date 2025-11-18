const term = document.getElementById("terminal");
const ws = new WebSocket(`ws://${location.host}/ws`);

ws.onmessage = (event) => {
  term.textContent += event.data + "\n";
  term.scrollTop = term.scrollHeight;
};
