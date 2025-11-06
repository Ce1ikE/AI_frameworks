from typing import List
from ..dataclasses import FaceEmbeddingMessage
from numpy import ndarray
from enum import Enum
from abc import abstractmethod
from ..utils.model_backend import BackendMixin, BackendType

class FaceClassifier(BackendMixin):
    task: str = "face_classification"

    def __init__(self, backend: BackendType):
        super().__init__()

        self.backend = backend

    @abstractmethod
    def predict(self, embedding: ndarray) -> str:
        """Return a label for the given embedding"""
        raise NotImplementedError("Predict method must be implemented by the classifier")
    
    def settings(self) -> dict:
        """Optional training metrics (inertia, silhouette, etc.)."""
        return {}
    
