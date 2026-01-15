const term = document.getElementById("terminal");
const wsStatus = document.getElementById("ws-status");
const lastUpdate = document.getElementById("last-update");
const clearButton = document.getElementById("clear-terminal");
const pauseButton = document.getElementById("pause-terminal");
const watchlist = document.getElementById("watchlist");
const positions = document.getElementById("positions");
const marketOpen = document.getElementById("market-open");
const tradingWindow = document.getElementById("trading-window");
const regimesBox = document.getElementById("regimes");

let paused = false;
let ws;

const appendLine = (message, type = "system") => {
  const line = document.createElement("div");
  line.className = `line ${type}`;
  line.textContent = message;
  term.appendChild(line);
  term.scrollTop = term.scrollHeight;
};

const setStatus = (text, statusClass) => {
  wsStatus.textContent = text;
  wsStatus.className = `pill ${statusClass}`;
};

const updateTimestamp = () => {
  const now = new Date();
  lastUpdate.textContent = now.toLocaleTimeString();
  lastUpdate.className = "pill ok";
};

const renderStatus = (payload) => {
  const symbols = payload.symbols || [];
  const openPositions = payload.open_positions || [];
  const regimes = payload.regimes || {};

  watchlist.textContent = symbols.length ? symbols.join(", ") : "--";
  positions.textContent = openPositions.length ? openPositions.join(", ") : "none";
  marketOpen.textContent = payload.market_open ? "Yes" : "No";
  tradingWindow.textContent = payload.within_trading_hours ? "Yes" : "No";

  const regimeLines = Object.entries(regimes)
    .map(([symbol, regime]) => `${symbol}: ${regime}`)
    .join("\n");
  regimesBox.textContent = regimeLines || "--";
};

const fetchStatus = async () => {
  try {
    const response = await fetch("/status");
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    renderStatus(payload);
  } catch (error) {
    // Ignore status fetch errors; websocket will still render.
  }
};

const connect = () => {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws`);

  setStatus("connecting", "warn");

  ws.onopen = () => {
    setStatus("connected", "ok");
    appendLine(">>> WebSocket connected. Streaming engine cycles...", "system");
  };

  ws.onmessage = (event) => {
    if (paused) {
      return;
    }
    appendLine(event.data, event.data.startsWith("ERROR") ? "error" : "system");
    updateTimestamp();
  };

  ws.onclose = () => {
    setStatus("disconnected", "warn");
    appendLine(">>> WebSocket connection closed. Reconnecting...", "system");
    setTimeout(connect, 3000);
  };

  ws.onerror = () => {
    setStatus("error", "warn");
  };
};

clearButton.addEventListener("click", () => {
  term.innerHTML = "";
  appendLine(">>> Terminal cleared.", "system");
});

pauseButton.addEventListener("click", () => {
  paused = !paused;
  pauseButton.textContent = paused ? "Resume" : "Pause";
  appendLine(paused ? ">>> Output paused." : ">>> Output resumed.", "system");
});

connect();
fetchStatus();
setInterval(fetchStatus, 30000);
