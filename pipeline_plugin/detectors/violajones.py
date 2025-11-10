import logging
from enum import Enum
import cv2
import numpy as np

from .base import FaceDetector
from ..dataclasses import ImageMessage
from ..utils.model_store import verify_model_weights

class CascadeType(Enum):
    FRONTALFACE_DEFAULT = "haarcascade_frontalface_default.xml"
    FRONTALFACE_ALT = "haarcascade_frontalface_alt.xml"
    FULLBODY = "haarcascade_fullbody.xml"
    EYE = "haarcascade_eye.xml"
    SMILE = "haarcascade_smile.xml"

    __all__ = ["FRONTALFACE_DEFAULT", "FRONTALFACE_ALT", "FULLBODY", "EYE", "SMILE"]

HAARCASCADE_URLS = {
    CascadeType.FRONTALFACE_DEFAULT: 'https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/haarcascade_frontalface_default.xml',
    CascadeType.FRONTALFACE_ALT: 'https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/haarcascade_frontalface_alt.xml',
    CascadeType.FULLBODY: 'https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/data/haarcascades/haarcascade_fullbody.xml',
    CascadeType.EYE: 'https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml',
    CascadeType.SMILE: 'https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml',
}

HAARCASCADE_SHA256 = {
    CascadeType.FRONTALFACE_DEFAULT: '0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0',
    CascadeType.FRONTALFACE_ALT: '6281df13459cc218ff047d02b2ae3859b12ff14a93ffe8952f7b33fad7b9697b',  
    CascadeType.FULLBODY: '041745c71eef1b5c86aef224f17ce75b042d33314cc8f6757424f8bd8cd30aa1',     
    CascadeType.EYE: '94749780bc646f6f172f827666f7a391aad9363fdb57d66f08f7fde7cdd067ec',
    CascadeType.SMILE: 'd70ce87df4d0c44552c1208831d4c91d31b697f6fd4b2a7a183a9550233b557c',
}

CHUNK_SIZE = 8192

class ViolaJonesDetector(FaceDetector):
    def __init__(
        self,
        model_dir: str, 
        model_name: CascadeType = CascadeType.FRONTALFACE_DEFAULT,
        input_size: int = (640, 640),
    ):
        super().__init__()
        self.cascade_path = verify_model_weights(
            model_name=model_name,
            root=model_dir / "violajones",
            model_urls=HAARCASCADE_URLS,
            model_sha256=HAARCASCADE_SHA256,
            chunk_size=CHUNK_SIZE
        )
        self.input_size = input_size
        self.model_name = model_name

    def detect_faces(self, image: ImageMessage):
        orig_h, orig_w = image.image.shape[:2]
        input_h, input_w = self.input_size[:2]

        gray = cv2.cvtColor(image.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.input_size)
        bboxes = cv2.CascadeClassifier(self.cascade_path).detectMultiScale(gray)

        if len(bboxes) == 0:
            print("No faces detected.")
            return (
                np.zeros((0, 4), dtype=np.float32),  
                np.zeros((0,), dtype=np.float32),    
                np.zeros((0, 5, 2), dtype=np.float32) 
            )

        bboxes = np.array([
            [x, y, x + w, y + h] for (x, y, w, h) in bboxes
        ], dtype=np.float32)

        # no scores as Viola-Jones does not provide confidence scores
        scores = None
        # no landmarks from Viola-Jones
        landmarks = None
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y

        print(f"Detected : {len(bboxes)} face(s)")
        return bboxes, scores, landmarks

    def settings(self):
        return {
            **super().settings(),
            "model_name": self.model_name.value,
            "opencv_version": cv2.__version__,
        }