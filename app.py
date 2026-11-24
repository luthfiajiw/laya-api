import inspect
import os
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TORCH_NUM_THREADS", "2")

import laya
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.getenv("LAYA_MODEL_PATH", "multilingual")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "state": "Kami ditagih dua kali untuk tagihan bulan Maret. Tolong segera refund hari ini juga.",
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Departemen mana yang harus menangani pesan ini?",
                        "criteria": {
                            "billing": "tagihan, pembayaran, refund",
                            "technical": "bug, error sistem, gangguan",
                            "sales": "harga, kontrak baru",
                        },
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "Seberapa mendesak permintaan ini?",
                        "criteria": [
                            "tidak mendesak",
                            "segera",
                            "sangat mendesak atau memblokir operasional",
                        ],
                    },
                    "refund_requested": {
                        "type": "noul",
                        "instructions": "Apakah pengguna meminta pengembalian dana?",
                    },
                },
                "min_confidence": 0.85,
            }
        }
    )

    state: Optional[Union[str, Dict[str, Any], List[Any]]] = Field(
        default=None,
        description="Input resmi Laya (teks, JSON object/dict, atau list)",
    )
    text: Optional[str] = Field(default=None, description="Alias opsional untuk state berupa teks")
    questions: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        description="Daftar atau mapping pertanyaan Laya (mendukung tipe choice, score, noul)"
    )
    min_confidence: Optional[Union[float, Dict[str, float]]] = Field(
        default=None,
        description="Batas minimum confidence Laya (float 0..1 atau dict per bucket) untuk menandai low_confidence",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Alias opsional untuk min_confidence jika min_confidence tidak diisi",
    )
    model: Optional[str] = Field(
        default=None,
        description="Override model/checkpoint opsional (juga untuk kompatibilitas payload ala Jev)",
    )
    lang: Optional[str] = Field(default=None, description="Hint bahasa opsional untuk Laya")
    max_len: Optional[int] = Field(default=None, ge=1, description="Batas panjang token state opsional")
    head_max_len: Optional[int] = Field(default=None, ge=1, description="Batas panjang token head/opsi opsional")


def build_predict_kwargs(predict_fn: Any, req: PredictRequest) -> Dict[str, Any]:
    effective_min_confidence = (
        req.min_confidence if req.min_confidence is not None else req.threshold
    )
    candidates = {
        "min_confidence": effective_min_confidence,
        "model": req.model,
        "lang": req.lang,
        "max_len": req.max_len,
        "head_max_len": req.head_max_len,
    }
    provided = {k: v for k, v in candidates.items() if v is not None}
    if not provided:
        return {}

    try:
        sig = inspect.signature(predict_fn)
    except (TypeError, ValueError):
        return {}

    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return provided
    return {k: v for k, v in provided.items() if k in sig.parameters}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "loaded": agent is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Model belum siap")

    input_state = req.state if req.state is not None else req.text
    is_empty_str = isinstance(input_state, str) and not input_state.strip()
    is_empty_collection = isinstance(input_state, (dict, list)) and len(input_state) == 0
    if input_state is None or is_empty_str or is_empty_collection:
        raise HTTPException(
            status_code=422,
            detail="Field 'state' (atau 'text') wajib diisi dan tidak boleh kosong",
        )

    if len(req.questions) == 0:
        raise HTTPException(
            status_code=422,
            detail="Field 'questions' wajib diisi dan tidak boleh kosong",
        )

    predict_kwargs = build_predict_kwargs(agent.predict, req)

    try:
        raw_result = agent.predict(input_state, req.questions, **predict_kwargs)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail={"error": str(exc), "exception": type(exc).__name__},
        )

    safe_raw = to_json_safe(raw_result)
    return {"response": safe_raw}
