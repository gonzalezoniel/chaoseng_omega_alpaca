import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger("chaoseng")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

engine = None
engine_error = None
_scheduler_task = None
_cycle_lock = asyncio.Lock()

SCHEDULER_INTERVAL = int(os.getenv("SCHEDULER_INTERVAL_SECONDS", "60"))


def _init_engine():
    global engine, engine_error
    try:
        from chaoseng.engine import ChaosEngineOmegaHybrid
        engine = ChaosEngineOmegaHybrid()
        logger.info("Engine initialized — paper=%s symbols=%s", engine.alpaca.is_paper(), engine.cfg.symbols)
    except Exception as exc:
        engine_error = str(exc)
        logger.warning("Engine unavailable: %s", engine_error)


async def _scheduler_loop():
    while True:
        if engine is not None:
            async with _cycle_lock:
                try:
                    msg = await engine.live_step()
                    logger.info("[Scheduler] %s", msg.replace("\n", " | "))
                except Exception as exc:
                    logger.error("[Scheduler] cycle error: %s", exc)
        await asyncio.sleep(SCHEDULER_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler_task
    _init_engine()
    if engine is not None:
        _scheduler_task = asyncio.create_task(_scheduler_loop())
        logger.info("Background scheduler started (interval=%ds)", SCHEDULER_INTERVAL)
    yield
    if _scheduler_task is not None:
        _scheduler_task.cancel()


app = FastAPI(title="Chaos Engine OMEGA", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_engine():
    if engine is None:
        raise RuntimeError(engine_error or "Chaos engine unavailable.")
    return engine


@app.get("/")
def root():
    if engine is None:
        return {
            "status": "degraded",
            "message": "Chaos Engine OMEGA — engine unavailable",
            "error": engine_error,
            "fix": "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables and redeploy.",
        }
    return {"status": "ok", "message": "Chaos Engine OMEGA Alpaca Hybrid Ready"}


@app.get("/health")
def health():
    return {
        "status": "ok" if engine is not None else "degraded",
        "engine_ready": engine is not None,
        "error": engine_error,
    }


@app.get("/dashboard")
async def dashboard(request: Request):
    pnl_data = None
    status_data = None
    history_data = None
    if engine is not None:
        try:
            pnl_data = engine.get_pnl_summary()
        except Exception:
            pass
        try:
            status_data = engine.get_status()
        except Exception:
            pass
        try:
            history_data = engine.get_trade_history(limit=20)
        except Exception:
            pass
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "engine_error": engine_error,
            "pnl": pnl_data,
            "status": status_data,
            "history": history_data,
        },
    )


@app.get("/history")
def history(limit: int = 200):
    try:
        return get_engine().get_trade_history(limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/status")
def status():
    try:
        return get_engine().get_status()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/pnl")
def pnl():
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
        async with _cycle_lock:
            try:
                msg = await engine.live_step()
            except Exception as e:
                msg = f"ERROR in live_step: {e}"
        await ws.send_text(msg)
        await asyncio.sleep(SCHEDULER_INTERVAL)
