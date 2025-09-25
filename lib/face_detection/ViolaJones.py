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

HAARCASCADE_URLS = {
    CascadeType.FRONTALFACE_DEFAULT: 'https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/haarcascade_frontalface_default.xml',
    CascadeType.FRONTALFACE_ALT: 'https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/haarcascade_frontalface_alt.xml',
    CascadeType.FULLBODY: 'https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/haarcascade_fullbody.xml',
}

HAARCASCADE_SHA256 = {
    CascadeType.FRONTALFACE_DEFAULT: '0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0',
    CascadeType.FRONTALFACE_ALT: '6281df13459cc218ff047d02b2ae3859b12ff14a93ffe8952f7b33fad7b9697b',  
    CascadeType.FULLBODY: '041745c71eef1b5c86aef224f17ce75b042d33314cc8f6757424f8bd8cd30aa1',     
}

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

    def detect_faces(self, image):
        self.bboxes = cv2.CascadeClassifier(self.cascade_path).detectMultiScale(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
        )
        self.logger.debug(f"Detected {len(self.bboxes)} face(s)")
        return self.bboxes , None , None

    def settings(self):
        return {
            "model_name": self.model_name,
            "model": self.get_name(),
            "opencv_version": cv2.__version__,
            "cascade_path": self.cascade_path.stem,
        }