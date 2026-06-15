from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from app.inference import run_inference
from app.schemas import PredictResponse

app = FastAPI(title="Brain MRI Tumor Inference API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
):
    try:
        result = await run_inference(
            files={
                "t1": t1,
                "t1ce": t1ce,
                "t2": t2,
                "flair": flair,
            }
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
