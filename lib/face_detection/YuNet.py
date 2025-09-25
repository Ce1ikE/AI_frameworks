from enum import Enum
from ..API.FaceDetector import FaceDetector
from cv2.typing import MatLike 
import cv2
import logging
import pprint
import numpy as np
from typing import Dict
from ..API.model_store import verify_model_weights

class YuNetWeights(str, Enum):
    YUNET = "yunet.onnx"

MODEL_URLS: Dict[YuNetWeights, str] = {
    YuNetWeights.YUNET: 'https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx?download=true',
}

MODEL_SHA256: Dict[YuNetWeights, str] = {
    YuNetWeights.YUNET: '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
}
CHUNK_SIZE = 8192

class YuNetDetector(FaceDetector):
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        model_name: YuNetWeights = YuNetWeights.YUNET,
    ):
        super().__init__(__class__.__name__)

        model_path = verify_model_weights(
            model_name, 
            model_urls=MODEL_URLS, 
            model_sha256=MODEL_SHA256,
            root="~/.yunet/weights",
            chunk_size=CHUNK_SIZE
        )
        self.detector = cv2.FaceDetectorYN.create(
            model_path, 
            "", 
            (640, 640)
        )
        self.detections = None
        self.model_name = model_name

    def detect_faces(self, image: MatLike):
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
        return bboxes , None , None


    def settings(self):
        return {
            "model_name": self.get_name(),
            "model": self.model_name.value,
            "opencv_version": cv2.__version__,
            "input_size": self.detector.getInputSize(),
            "nms_threshold": str(self.detector.getNMSThreshold()),
            "score_threshold": str(self.detector.getScoreThreshold()),
            "top_k": str(self.detector.getTopK()),
        }