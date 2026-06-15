from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import numpy as np
import pydicom
from PIL import Image

app = FastAPI()

UPLOAD_DIR = "uploads"
PREVIEW_DIR = "previews"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

app.mount("/previews", StaticFiles(directory=PREVIEW_DIR), name="previews")


@app.get("/")
def root():
    return {"message": "MRI DICOM Segmentation API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename.lower()

    if not filename.endswith(".dcm"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only DICOM .dcm files are allowed"}
        )

    unique_id = uuid.uuid4().hex
    dicom_filename = f"{unique_id}.dcm"
    dicom_path = os.path.join(UPLOAD_DIR, dicom_filename)

    contents = await file.read()
    with open(dicom_path, "wb") as f:
        f.write(contents)

    try:
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid DICOM file: {str(e)}"}
        )

    min_val = pixel_array.min()
    max_val = pixel_array.max()

    if max_val > min_val:
        normalized = (pixel_array - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(pixel_array)

    image_8bit = (normalized * 255).astype(np.uint8)

    if image_8bit.ndim == 3:
        preview_slice = image_8bit[image_8bit.shape[0] // 2]
    else:
        preview_slice = image_8bit

    preview_filename = f"{unique_id}.png"
    preview_path = os.path.join(PREVIEW_DIR, preview_filename)

    preview_image = Image.fromarray(preview_slice)
    preview_image.save(preview_path)

    return JSONResponse(content={
        "message": "DICOM received successfully",
        "original_filename": file.filename,
        "saved_dicom": dicom_filename,
        "preview_url": f"/previews/{preview_filename}",
        "shape": list(pixel_array.shape),
        "dtype": str(pixel_array.dtype),
        "prediction": {
            "label": "demo_segmentation",
            "confidence": 0.95,
            "tumor_detected": True
        }
    })
