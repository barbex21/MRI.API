from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="Brain MRI Tumor Inference API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
):
    try:
        required_files = {
            "t1": t1,
            "t1ce": t1ce,
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

        return {
            "status": "ok",
            "message": "Predict endpoint is available",
            "received_files": {
                "t1": t1.filename,
                "t1ce": t1ce.filename,
                "t2": t2.filename,
                "flair": flair.filename,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
