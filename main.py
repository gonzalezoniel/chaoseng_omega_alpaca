from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from chaoseng.engine import ChaosEngineOmegaHybrid
import asyncio

app = FastAPI()

# Serve /static for CSS + JS
app.mount("/static", StaticFiles(directory="static"), name="static")

engine = ChaosEngineOmegaHybrid()


@app.get("/")
def root():
    return {"message": "Chaos Engine OMEGA Alpaca Hybrid Ready"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/dashboard")
async def dashboard():
    try:
        engine.run_cycle()
        return HTMLResponse("""
        >>> Connected to Chaos Engine OMEGA terminal.<br>
        Waiting for next cycle...<br>
        <span style="color: #00FF00;">Engine cycle executed successfully.</span>
        """)
    except Exception as e:
        return HTMLResponse(f"""
        ERROR in cycle:<br>
        <span style="color: #FF4444;">{str(e)}</span>
        """)


@app.get("/history")
def history(limit: int = 200):
    """
    Return recent trade events (ENTER / EXIT) as JSON.
    """
    return engine.get_trade_history(limit=limit)


@app.get("/pnl")
def pnl():
    """
    Simple realized PnL summary.
    """
    return engine.get_pnl_summary()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text(">>> Connected to Chaos Engine OMEGA terminal. Waiting for first cycle...\n")

    while True:
        try:
            msg = await engine.live_step()
        except Exception as e:
            msg = f"ERROR in live_step: {e}"
        await ws.send_text(msg)
        await asyncio.sleep(60)  # 1-minute cycle
