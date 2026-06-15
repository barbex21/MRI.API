import os
from pathlib import Path
import numpy as np
import nibabel as nib
import torch

from fastapi import HTTPException
from model import UNet3D


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model (3).pth"
CASES_DIR = BASE_DIR / "cases"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (96, 96, 96)

model = UNet3D(in_channels=4, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def center_crop_3d(data: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
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


def find_case_files(case_id: str) -> dict:
    candidates = {
        "t1": [
            CASES_DIR / f"{case_id}_T1.nii.gz",
            CASES_DIR / f"{case_id}_t1.nii.gz",
        ],
        "t1gd": [
            CASES_DIR / f"{case_id}_T1GD.nii.gz",
            CASES_DIR / f"{case_id}_T1Gd.nii.gz",
            CASES_DIR / f"{case_id}_t1gd.nii.gz",
        ],
        "t2": [
            CASES_DIR / f"{case_id}_T2.nii.gz",
            CASES_DIR / f"{case_id}_t2.nii.gz",
        ],
        "flair": [
            CASES_DIR / f"{case_id}_FLAIR.nii.gz",
            CASES_DIR / f"{case_id}_flair.nii.gz",
        ],
    }

    resolved = {}

    for modality, options in candidates.items():
        match = next((p for p in options if p.exists()), None)
        if match is None:
            raise HTTPException(
                status_code=404,
                detail=f"Missing file for modality '{modality}' in cases folder for case_id '{case_id}'."
            )
        resolved[modality] = match

    return resolved


def load_nifti_volume(path: Path) -> np.ndarray:
    try:
        img = nib.load(str(path))
        data = img.get_fdata().astype(np.float32)
        return data
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read NIfTI file: {path.name}"
        )


def run_inference_by_case(case_id: str) -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model weights not found: {MODEL_PATH.name}"
        )

    files = find_case_files(case_id)

    imgs = [
        load_nifti_volume(files["t1"]),
        load_nifti_volume(files["t1gd"]),
        load_nifti_volume(files["t2"]),
        load_nifti_volume(files["flair"]),
    ]

    base_shape = imgs[0].shape
    for img in imgs[1:]:
        if img.shape != base_shape:
            raise HTTPException(
                status_code=400,
                detail=f"All modalities must have the same shape. Expected {base_shape}."
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

    return {
        "tumor_detected": bool(mask_np.sum() > 0),
        "tumor_voxel_count": int(mask_np.sum()),
        "mean_probability": float(probs_np.mean()),
        "max_probability": float(probs_np.max()),
        "input_shape": list(vol.shape),
        "cropped_shape": list(vol_c.shape),
        "used_files": {k: str(v.name) for k, v in files.items()},
        "note": "Inference completed successfully from server-side case files."
    }
