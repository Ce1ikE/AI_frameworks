from enum import Enum
from .base import FaceDetector
import cv2
import logging
from typing import Dict
import numpy as np

from uniface.common import (
    nms,
    resize_image,
)


from ..dataclasses import ImageMessage
from ..utils.model_store import verify_model_weights
from ..utils.util_functions import Utils

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
        model_dir: str,
        model_name: YuNetWeights = YuNetWeights.YUNET,
        input_size: tuple[int, int] = (640,640),
        nms_threshold: float = 0.4,
        confidence_threshold: float = 0.5,
        top_k: int = 5000,
    ):
        super().__init__(__class__.__name__)

        self.model_path = verify_model_weights(
            model_name, 
            model_urls=MODEL_URLS, 
            model_sha256=MODEL_SHA256,
            root=model_dir / "yunet",
            chunk_size=CHUNK_SIZE
        )
        
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.detections = None
        self.model_name = model_name
        self.input_size = input_size
        self.detector = None
        self.initialized = False

    def _lazy_init(self):
        if not self.initialized:
            self.detector = cv2.FaceDetectorYN.create(
                self.model_path, 
                "", 
                self.input_size
            )
            self.detector.setNMSThreshold(self.nms_threshold)
            self.detector.setScoreThreshold(self.confidence_threshold)
            self.detector.setTopK(self.top_k)
            self.detector.setInputSize(self.input_size)
            self.initialized = True

    def detect_faces(self, message: ImageMessage):
        self._lazy_init()
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
        image = message.image

        orig_image = message.image
        orig_h, orig_w = orig_image.shape[:2]

        # Resize for inference
        if self.input_size is not None:
            input_h, input_w = self.input_size[1], self.input_size[0]
            image = cv2.resize(orig_image, self.input_size)
        else:
            input_h, input_w = orig_h, orig_w
            self.detector.setInputSize((orig_w, orig_h))
            image = orig_image

        n_faces, self.detections = self.detector.detect(image)
        if n_faces == 0:
            self.logger.info("No faces detected.")
            return None, None, None
        
        bboxes = []
        landmarks = []
        scores = []

        for detection in self.detections:
            x, y, w, h = detection[:4].astype(int)
            score = float(detection[14])
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            landmark = detection[4:14].reshape((5, 2)).astype(np.float32)

            bboxes.append(bbox)
            landmarks.append(landmark)
            scores.append(score)

        bboxes = np.stack(bboxes).astype(np.float32)
        landmarks = np.stack(landmarks).astype(np.float32) 
        scores = np.array(scores, dtype=np.float32)

        input_h, input_w = self.input_size[1], self.input_size[0]
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y
        landmarks[:, :, 0] *= scale_x
        landmarks[:, :, 1] *= scale_y

        print(f"Detected : {len(bboxes)} face(s) with score(s) {scores}")
        print(f"Detected : At {bboxes}")
        return bboxes, scores, landmarks


    def settings(self):
        return {
            "model_name": self.model_name.value,
            "model": self.model_name.value,
            "opencv_version": cv2.__version__,
            "input_size": self.input_size,
            "nms_threshold": str(self.detector.getNMSThreshold()),
            "score_threshold": str(self.detector.getScoreThreshold()),
            "top_k": str(self.detector.getTopK()),
        }