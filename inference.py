from fastapi import UploadFile, HTTPException


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}


def _has_allowed_extension(filename: str) -> bool:
    filename = filename.lower()
    return any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)


async def run_inference(
    t1: UploadFile,
    t1ce: UploadFile,
    t2: UploadFile,
    flair: UploadFile,
):
    files = {
        "t1": t1,
        "t1ce": t1ce,
        "t2": t2,
        "flair": flair,
    }

    for modality, file in files.items():
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail=f"Missing filename for {modality}"
            )

        if not _has_allowed_extension(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for {modality}: {file.filename}"
            )

    return {
        "status": "ok",
        "message": "Dummy inference completed successfully",
        "prediction": "no_model_yet",
        "received_files": {
            "t1": t1.filename,
            "t1ce": t1ce.filename,
            "t2": t2.filename,
            "flair": flair.filename,
        }
    }
