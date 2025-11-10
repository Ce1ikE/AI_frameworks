from ..dataclasses import FaceMessage, FaceEmbeddingMessage
from abc import abstractmethod
import numpy as np

class FaceEmbedder:
    task: str = "face_embedding"

    def __init__(self):
        pass

    @abstractmethod
    def embed_face(self, face: FaceMessage) -> np.ndarray:
        """Return an embedding vector (e.g. 512-D)"""
        raise NotImplementedError
    
    def settings(self) -> dict:
        return {
            "task": self.task
        }
