from typing import List
from ..dataclasses import FaceEmbeddingMessage
from numpy import ndarray
from enum import Enum
from abc import abstractmethod
from ..utils.model_backend import BackendMixin, BackendType

class LearningType(Enum):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"

class FaceClassifier(BackendMixin):
    task: str = "face_classification"

    def __init__(self, backend: BackendType, learning_type: LearningType):
        super().__init__()

        self.backend = backend
        self.learning_type = learning_type

    @abstractmethod
    def predict(self, embedding: ndarray) -> str:
        """Return a label for the given embedding"""
        raise NotImplementedError("Predict method must be implemented by the classifier")
    
    @abstractmethod
    def train(self,embeddings: ndarray,labels: List[str] = None) -> None:
        """Train the classifier. Unsupervised models can ignore labels"""
        raise NotImplementedError

    def metrics(self) -> dict:
        """Optional training metrics (inertia, silhouette, etc.)."""
        return {}