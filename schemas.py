from pydantic import BaseModel
from typing import Dict, List, Optional


class HealthResponse(BaseModel):
    status: str
    message: str


class PredictionResult(BaseModel):
    tumor_detected: bool
    tumor_voxel_count: int
    mean_probability: float
    max_probability: float
    input_shape: List[int]
    cropped_shape: List[int]
    note: Optional[str] = None


class PredictResponse(BaseModel):
    status: str
    message: str
    received_files: Dict[str, str]
    result: PredictionResult
