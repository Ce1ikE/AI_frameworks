from ..dataclasses import ImageDetectionMessage, ImageMessage
from ..utils.model_backend import BackendMixin, BackendType
from abc import ABC,abstractmethod
import numpy as np

class FaceDetector(BackendMixin):
    task: str = "face_detection"

    def __init__(self, backend: BackendType = BackendType.OPENCV):
        self.backend = backend
        super().__init__()

    @abstractmethod
    def detect_faces(self, image: ImageMessage) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, None] | tuple[np.ndarray, None, None]:
        """Return a list of bounding boxes [(x, y, w, h), ...] and optionally landmarks and scores"""
        raise NotImplementedError("detect_faces method must be implemented by subclasses")

    def settings(self) -> dict:
        return {
            "backend": self.backend.value,
            "task": self.task
        }