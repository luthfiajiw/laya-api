import os
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TORCH_NUM_THREADS", "2")

import laya
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("LAYA_MODEL_PATH", "multilingual")
agent = None


def to_json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "item"):  # Tangani tipe numerik NumPy (float32/int64)
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return to_json_safe(vars(obj))
    return str(obj)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = laya.load(MODEL_PATH)
    yield
    agent = None


app = FastAPI(title="Laya Decision Service", lifespan=lifespan)


class PredictRequest(BaseModel):
    state: Optional[str] = Field(default=None, description="Teks input resmi Laya")
    text: Optional[str] = Field(default=None, description="Alias opsional untuk state")
    questions: Union[Dict[str, Any], List[Dict[str, Any]]]
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "loaded": agent is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Model belum siap")

    input_state = req.state if req.state is not None else req.text
    if not input_state or not input_state.strip():
        raise HTTPException(
            status_code=422,
            detail="Field 'state' (atau 'text') wajib diisi dan tidak boleh kosong",
        )

    try:
        raw_result = agent.predict(input_state, req.questions)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail={"error": str(exc), "exception": type(exc).__name__},
        )

    safe_raw = to_json_safe(raw_result)
    answers = safe_raw.get("answers", safe_raw) if isinstance(safe_raw, dict) else {}

    return {"response": safe_raw}
