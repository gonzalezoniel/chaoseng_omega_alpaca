const term = document.getElementById("terminal");

// Use wss:// on https, ws:// on http
const protocol = location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${protocol}://${location.host}/ws`);

ws.onmessage = (event) => {
  term.textContent += event.data + "\n";
  term.scrollTop = term.scrollHeight;
};

ws.onclose = () => {
  term.textContent += "\n>>> WebSocket connection closed.\n";
};