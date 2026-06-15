from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import uuid

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "MRI Segmentation API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Only PNG and JPG files are allowed"}
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file"}
        )

    width, height = image.size

    ext = ".png" if file.content_type == "image/png" else ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    image.save(save_path)

    return JSONResponse(content={
        "message": "Image received successfully",
        "filename": file.filename,
        "saved_as": unique_name,
        "content_type": file.content_type,
        "width": width,
        "height": height,
        "prediction": {
            "label": "demo_segmentation",
            "confidence": 0.95,
            "tumor_detected": True
        }
    })
