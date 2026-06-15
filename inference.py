import os
import uuid
import torch
import numpy as np
import nibabel as nib
from fastapi import UploadFile
from app.model import UNet3D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "weights/bestmodel.pth"
TARGET_SIZE = (96, 96, 96)

model = UNet3D(in_channels=4, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def center_crop_3d(data, target_shape):
    start = [(d - t) // 2 for d, t in zip(data.shape[-3:], target_shape)]
    return data[
        ...,
        start[0]:start[0] + target_shape[0],
        start[1]:start[1] + target_shape[1],
        start[2]:start[2] + target_shape[2],
    ]

def normalize_channels(vol):
    vol = vol.copy()
    for i in range(4):
        vol[i] = (vol[i] - vol[i].mean()) / (vol[i].std() + 1e-8)
    return vol

async def save_upload(upload: UploadFile, folder="uploads"):
    os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(upload.filename)[1] or ".nii.gz"
    path = os.path.join(folder, f"{uuid.uuid4().hex}{ext}")
    content = await upload.read()
    with open(path, "wb") as f:
        f.write(content)
    return path

async def run_inference(files: dict):
    paths = {}
    for key, upload in files.items():
        paths[key] = await save_upload(upload)

    imgs = [
        nib.load(paths["t1"]).get_fdata().astype(np.float32),
        nib.load(paths["t1ce"]).get_fdata().astype(np.float32),
        nib.load(paths["t2"]).get_fdata().astype(np.float32),
        nib.load(paths["flair"]).get_fdata().astype(np.float32),
    ]

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
    }
