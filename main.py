from fastapi import FastAPI, UploadFile, File, HTTPException
from inference import run_inference
from schemas import HealthResponse, PredictResponse

app = FastAPI(title="Brain MRI Tumor Inference API")


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "message": "API is running"
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    t1: UploadFile = File(...),
    t1gd: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
):
    required_files = {
        "t1": t1,
        "t1gd": t1gd,
        "t2": t2,
        "flair": flair,
    }

    missing_files = [
        name for name, uploaded_file in required_files.items()
        if not uploaded_file.filename
    ]

    if missing_files:
        raise HTTPException(
            status_code=400,
            detail=f"Missing files: {', '.join(missing_files)}"
        )

    try:
        result = await run_inference(required_files)

        return {
            "status": "ok",
            "message": "Inference completed successfully",
            "received_files": {
                "t1": t1.filename,
                "t1gd": t1gd.filename,
                "t2": t2.filename,
                "flair": flair.filename,
            },
            "result": result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
