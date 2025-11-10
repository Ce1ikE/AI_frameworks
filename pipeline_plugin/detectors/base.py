from ..dataclasses import ImageDetectionMessage, ImageMessage
from abc import ABC,abstractmethod
import numpy as np

class FaceDetector:
    task: str = "face_detection"

    def __init__(self):
        pass

    @abstractmethod
    def detect_faces(self, image):
        raise NotImplementedError("detect_faces method must be implemented by subclasses")

    def settings(self) -> dict:
        return {
            "task": self.task
        }