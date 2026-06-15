from pydantic import BaseModel
from typing import List

class PredictResponse(BaseModel):
    tumor_detected: bool
    tumor_voxel_count: int
    mean_probability: float
    max_probability: float
    input_shape: List[int]
    cropped_shape: List[int]
