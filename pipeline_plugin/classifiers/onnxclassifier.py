import logging
from .base import FaceClassifier
import numpy as np
from enum import Enum
from pathlib import Path
import onnxruntime as ort
from sklearn.metrics.pairwise import (
    cosine_similarity, 
    euclidean_distances, 
    manhattan_distances, 
    paired_distances
)

class ONNXClassifier(FaceClassifier):
    def __init__(
        self, 
        model_path: Path
    ):
        super().__init__()

    def predict(self, embedding: np.ndarray) -> int:
        pass
    
    def settings(self):
        return {
            "onnx_runtime_version": ort.__version__,
        }
