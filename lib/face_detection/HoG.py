from ..API.FaceDetector import FaceDetector
from cv2.typing import MatLike 
import logging
import cv2


class HoGDetector(FaceDetector):
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        model_name: str = None,
    ):
        super().__init__(__class__.__name__ if model_name is None else model_name)
        self.detector = cv2.HOGDescriptor()
    
    def detect_faces(self, image: MatLike) -> list[tuple[int, int, int, int]]:
        self.logger.debug("Detecting faces...")
        # Implement face detection logic here
        return []
