from pydantic import BaseModel
from typing import List, Dict, Union

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    class_name: str
    probability: float
    all_probabilities: Dict[str, float]