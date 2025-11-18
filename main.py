from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from chaoseng.engine import ChaosEngineOmegaHybrid
import asyncio

app = FastAPI()
engine = ChaosEngineOmegaHybrid()

@app.get("/dashboard")
def dashboard():
    with open("templates/dashboard.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # Simple loop: each iteration pulls data, makes decisions, streams log
    while True:
        msg = await engine.live_step()
        await ws.send_text(msg)
        await asyncio.sleep(60)  # 1-minute loop; adjust if needed

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Chaos Engine OMEGA Alpaca Hybrid Ready"}
