from pydantic import BaseModel
from typing import Dict, List, Optional


class HealthResponse(BaseModel):
    status: str
    message: str


class PredictByCaseRequest(BaseModel):
    case_id: str


class PredictionResult(BaseModel):
    tumor_detected: bool
    tumor_voxel_count: int
    mean_probability: float
    max_probability: float
    input_shape: List[int]
    cropped_shape: List[int]
    used_files: Dict[str, str]
    note: Optional[str] = None


class PredictResponse(BaseModel):
    status: str
    message: str
    case_id: str
    result: PredictionResult
