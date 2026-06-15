from fastapi import FastAPI, HTTPException
from inference import run_inference_by_case
from schemas import HealthResponse, PredictByCaseRequest, PredictResponse

app = FastAPI(title="Brain MRI Tumor Inference API")


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "message": "API is running"
    }


@app.post("/predict-by-case", response_model=PredictResponse)
def predict_by_case(payload: PredictByCaseRequest):
    try:
        result = run_inference_by_case(payload.case_id)

        return {
            "status": "ok",
            "message": "Inference completed successfully",
            "case_id": payload.case_id,
            "result": result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
