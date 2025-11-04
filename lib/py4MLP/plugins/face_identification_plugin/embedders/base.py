from ..dataclasses import FaceMessage, FaceEmbeddingMessage
from ..utils.model_backend import BackendMixin, BackendType
from abc import abstractmethod
import numpy as np

class FaceEmbedder(BackendMixin):
    task: str = "face_embedding"

    def __init__(self, backend: BackendType = BackendType.ONNX):
        self.backend = backend
        super().__init__()

    @abstractmethod
    def embed_face(self, face: FaceMessage) -> np.ndarray:
        """Return an embedding vector (e.g. 512-D)"""
        raise NotImplementedError
    
    def settings(self) -> dict:
        return {
            "backend": self.backend.value,
            "task": self.task
        }
