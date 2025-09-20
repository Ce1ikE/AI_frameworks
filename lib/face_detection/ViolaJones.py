# https://realpython.com/traditional-face-detection-python/
# https://www.youtube.com/watch?v=uEJ71VlUmMQ&t=491s
# https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
# https://jakevdp.github.io/PythonDataScienceHandbook/05.14-image-features.html

# haarcascade files can be found at the OpenCV repository:
# https://github.com/opencv/opencv/blob/master/data/haarcascades/

import logging
import cv2
from .CascadeType import CascadeType
from ..API.FaceDetector import FaceDetector
from PIL.Image import Image

class ViolaJones(FaceDetector):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        cascade_type: CascadeType = CascadeType.FRONTALFACE_DEFAULT
    ):
        self.cascade_type = cascade_type

    def detect_faces(self, image: Image) -> list[tuple[int,int,int,int]]:
        return cv2.CascadeClassifier(self.cascade_type.value).detectMultiScale(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
        )

    def detect_and_draw_faces(self, image: Image) -> tuple[list[tuple[int,int,int,int]], Image]:
        self.detected_faces = self.detect_faces(image)
        self.input_image = image.copy()
        self.logger.debug(f"Detected {len(self.detected_faces)} faces")
        for (x, y, width, height) in self.detected_faces:
            cv2.rectangle(
                self.input_image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                4
            )
        return (self.detected_faces, self.input_image)
