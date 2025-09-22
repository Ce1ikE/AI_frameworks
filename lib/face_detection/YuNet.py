from ..API.FaceDetector import FaceDetector
from cv2.typing import MatLike 
import cv2
import logging
import pprint
class YuNetDetector(FaceDetector):
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        model_path="yunet.onnx",
        model_name: str = None,
    ):
        super().__init__(__class__.__name__ if model_name is None else model_name)
        self.detector = cv2.FaceDetectorYN.create(model_path, "", (640, 640))
        self.detections = None

    def detect_faces(self, image: MatLike) -> list[tuple[int, int, int, int]]:
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        # output: 1xN x15 see: https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html#ac05bd075ca3e6edc0e328927aae6f45b
        # faces	detection results stored in a 2D cv::Mat of shape [num_faces, 15]
        # 0-1: x, y of bbox top left corner
        # 2-3: width, height of bbox
        # 4-5: x, y of right eye (blue point in the example image)
        # 6-7: x, y of left eye (red point in the example image)
        # 8-9: x, y of nose tip (green point in the example image)
        # 10-11: x, y of right corner of mouth (pink point in the example image)
        # 12-13: x, y of left corner of mouth (yellow point in the example image)
        # 14: face score
        _, self.detections = self.detector.detect(image)
        bboxes = []
        
        if self.detections is None:
            self.logger.info("No faces detected.")
            return bboxes
        
        for detection in self.detections:
            x, y, w, h = map(int, detection[0:4])
            bboxes.append((x, y, w, h))
        
        self.logger.info(f"""\n
            bboxes: {pprint.pformat(bboxes)},
        """)
        return bboxes
