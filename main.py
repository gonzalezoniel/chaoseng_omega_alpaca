from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from chaoseng.engine import ChaosEngineOmegaHybrid
import asyncio

app = FastAPI()

# Serve /static for CSS + JS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

engine = None
engine_error = None
try:
    engine = ChaosEngineOmegaHybrid()
except Exception as exc:
    engine_error = str(exc)


def get_engine() -> ChaosEngineOmegaHybrid:
    if engine is None:
        raise RuntimeError(engine_error or "Chaos engine unavailable.")
    return engine


@app.get("/")
def root():
    return {"message": "Chaos Engine OMEGA Alpaca Hybrid Ready"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "engine_error": engine_error,
        },
    )


@app.get("/history")
def history(limit: int = 200):
    """
    Return recent trade events (ENTER / EXIT) as JSON.
    """
    try:
        return get_engine().get_trade_history(limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/status")
def status():
    """
    Return engine status, watchlist, and last cycle metadata.
    """
    try:
        return get_engine().get_status()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/pnl")
def pnl():
    """
    Simple realized PnL summary.
    """
    try:
        return get_engine().get_pnl_summary()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    if engine is None:
        await ws.send_text(f">>> Engine unavailable: {engine_error}\n")
        await ws.close(code=1011)
        return
    await ws.send_text(">>> Connected to Chaos Engine OMEGA terminal. Waiting for first cycle...\n")

    while True:
        try:
            msg = await get_engine().live_step()
        except Exception as e:
            msg = f"ERROR in live_step: {e}"
        await ws.send_text(msg)
        await asyncio.sleep(60)  # 1-minute cycle
