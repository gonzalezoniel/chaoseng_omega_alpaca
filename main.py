from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from chaoseng.engine import ChaosEngineOmegaHybrid
import asyncio

app = FastAPI()

# Serve /static (CSS + JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

engine = ChaosEngineOmegaHybrid()


@app.get("/")
def root():
    return {"message": "Chaos Engine OMEGA Alpaca Hybrid Ready"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/dashboard")
def dashboard():
    # Simple file read for the terminal HTML
    with open("templates/dashboard.html") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # Initial message so you know the WS is connected
    await ws.send_text(">>> Connected to Chaos Engine OMEGA terminal. Waiting for first cycle...\n")

    while True:
        try:
            msg = await engine.live_step()
        except Exception as e:
            # Don't let one error kill the whole WebSocket
            msg = f"ERROR in live_step: {e}"
        await ws.send_text(msg)
        # 60 seconds per cycle (1-minute bars)
        await asyncio.sleep(60)