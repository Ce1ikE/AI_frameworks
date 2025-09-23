import logging
from pathlib import Path
from enum import Enum
import cv2
from ..API.FaceDetector import FaceDetector
from PIL.Image import Image

class CascadeType(Enum):
    FRONTALFACE_DEFAULT = "haarcascade_frontalface_default.xml"
    FRONTALFACE_ALT = "haarcascade_frontalface_alt.xml"
    FULLBODY = "haarcascade_fullbody.xml"

    __all__ = ["FRONTALFACE_DEFAULT", "FRONTALFACE_ALT", "FULLBODY"]

class ViolaJonesDetector(FaceDetector):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        cascade_path: Path,
        model_name: str = None
    ):
        super().__init__(__class__.__name__)
        self.model_name = model_name
        self.cascade_path = cascade_path

    def detect_faces(self, image: Image) -> list[tuple[int,int,int,int]]:
        self.bboxes = cv2.CascadeClassifier(self.cascade_path).detectMultiScale(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
        )
        self.logger.debug(f"Detected {len(self.bboxes)} face(s)")
        return self.bboxes

    def detect_and_draw_faces(self, image: Image) -> tuple[list[tuple[int,int,int,int]], Image]:
        self.bboxes = self.detect_faces(image)
        self.input_image = image.copy()
        for (x, y, width, height) in self.bboxes:
            cv2.rectangle(
                self.input_image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                4
            )
        return (self.bboxes, self.input_image)

    def settings(self):
        return {
            "model_name": self.model_name,
            "model": self.get_name(),
            "opencv_version": cv2.__version__,
            "cascade_path": self.cascade_path.stem,
        }