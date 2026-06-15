import os
import uuid
from pathlib import Path
import numpy as np
import nibabel as nib
import torch

from fastapi import UploadFile, HTTPException
from model import UNet3D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).resolve().parent / "best_model (3).pth"
TARGET_SIZE = (96, 96, 96)
UPLOAD_DIR = "uploads"

model = UNet3D(in_channels=4, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def is_allowed_nifti(filename: str) -> bool:
    filename = filename.lower()
    return filename.endswith(".nii") or filename.endswith(".nii.gz")


def center_crop_3d(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    spatial_shape = data.shape[-3:]

    if any(d < t for d, t in zip(spatial_shape, target_shape)):
        raise HTTPException(
            status_code=400,
            detail=f"Input volume is smaller than target crop size {target_shape}. Got {spatial_shape}."
        )

    start = [(d - t) // 2 for d, t in zip(spatial_shape, target_shape)]

    return data[
        ...,
        start[0]:start[0] + target_shape[0],
        start[1]:start[1] + target_shape[1],
        start[2]:start[2] + target_shape[2],
    ]


def normalize_channels(vol: np.ndarray) -> np.ndarray:
    vol = vol.copy()

    for i in range(vol.shape[0]):
        channel = vol[i]
        mean = channel.mean()
        std = channel.std()

        if std < 1e-8:
            vol[i] = channel - mean
        else:
            vol[i] = (channel - mean) / std

    return vol


async def save_upload(upload: UploadFile, folder: str = UPLOAD_DIR) -> str:
    os.makedirs(folder, exist_ok=True)

    if not upload.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    if not is_allowed_nifti(upload.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {upload.filename}. Only .nii and .nii.gz are allowed."
        )

    if upload.filename.lower().endswith(".nii.gz"):
        ext = ".nii.gz"
    else:
        ext = ".nii"

    path = os.path.join(folder, f"{uuid.uuid4().hex}{ext}")

    content = await upload.read()
    with open(path, "wb") as f:
        f.write(content)

    return path


def load_nifti_volume(path: str) -> np.ndarray:
    try:
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        return data
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read NIfTI file: {os.path.basename(path)}"
        )


async def run_inference(files: dict) -> dict:
    saved_paths = {}

    try:
        for key, upload in files.items():
            saved_paths[key] = await save_upload(upload)

        required_keys = ["t1", "t1gd", "t2", "flair"]
        missing_keys = [k for k in required_keys if k not in saved_paths]
        if missing_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required modalities: {', '.join(missing_keys)}"
            )

        imgs = [
            load_nifti_volume(saved_paths["t1"]),
            load_nifti_volume(saved_paths["t1gd"]),
            load_nifti_volume(saved_paths["t2"]),
            load_nifti_volume(saved_paths["flair"]),
        ]

        base_shape = imgs[0].shape
        for idx, img in enumerate(imgs[1:], start=1):
            if img.shape != base_shape:
                raise HTTPException(
                    status_code=400,
                    detail=f"All modalities must have the same shape. Expected {base_shape}, got {img.shape}."
                )

        vol = np.stack(imgs, axis=0)
        vol_c = center_crop_3d(vol, TARGET_SIZE)
        vol_n = normalize_channels(vol_c)

        x = torch.from_numpy(vol_n).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)
            mask = (probs > 0.5).float()

        mask_np = mask.squeeze().cpu().numpy()
        probs_np = probs.squeeze().cpu().numpy()

        tumor_detected = bool(mask_np.sum() > 0)
        tumor_voxel_count = int(mask_np.sum())
        mean_probability = float(probs_np.mean())
        max_probability = float(probs_np.max())

        return {
            "tumor_detected": tumor_detected,
            "tumor_voxel_count": tumor_voxel_count,
            "mean_probability": mean_probability,
            "max_probability": max_probability,
            "input_shape": list(vol.shape),
            "cropped_shape": list(vol_c.shape),
            "note": "Inference completed successfully"
        }

    finally:
        for path in saved_paths.values():
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
