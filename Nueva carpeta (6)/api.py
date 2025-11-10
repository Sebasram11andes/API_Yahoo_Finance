
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn, os, joblib, pandas as pd

# ==== Rutas ====
BASE_DIR = r"C:\Users\Leonardo\Documents\Nueva carpeta (6)"
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.joblib")

# ==== Cargar modelo (Pipeline: preprocesamiento + regresor) ====
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No se encontró el modelo en: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Columnas esperadas por el modelo (en el mismo orden en que entrenaste)
FEATURES = ["Ticker","Close","Volume","ret_1d","ma_5","ma_20","vol_ma5","rsi_14"]

# ==== FastAPI ====
app = FastAPI(title="Entrega2 - API de Predicción", version="1.0.0")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class PredictItem(BaseModel):
    records: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictItem):
    try:
        df = pd.DataFrame(payload.records)
        # Validar columnas
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")
        # Reordenar y predecir
        X = df[FEATURES].copy()
        yhat = model.predict(X)
        out = df.copy()
        out["prediction"] = yhat
        return {
            "n_records": len(out),
            "predictions": out[["prediction"]].round(6).to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
