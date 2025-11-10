from typing import List
from ..dataclasses import FaceEmbeddingMessage
from numpy import ndarray
from enum import Enum
from abc import abstractmethod

class FaceClassifier:
    task: str = "face_classification"

    def __init__(self):
        pass

    @abstractmethod
    def predict(self, embedding: ndarray) -> str:
        """Return a label for the given embedding"""
        raise NotImplementedError("Predict method must be implemented by the classifier")
    
    def settings(self) -> dict:
        """Optional training metrics (inertia, silhouette, etc.)."""
        return {}
    
