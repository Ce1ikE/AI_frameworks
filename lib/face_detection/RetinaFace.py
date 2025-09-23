from cv2.typing import MatLike 
import logging
import numpy as np
import pprint
import onnxruntime as ort
from ..API.Preprocessor import Preprocessor
from ..API.FaceDetector import FaceDetector
from uniface import RetinaFace
from uniface.constants import RetinaFaceWeights
# think of RetinaFace as:
# 1) place millions of anchors of various sizes on the image.
# 2) Model says:
# 2.1)"This anchor is background"
# 2.2)"This anchor should shift by (dx, dy, dw, dh) to tightly fit a face"
# 3) decode those offsets -> real face bounding box.
# 4) NMS (Non-Maximum Suppression) removes overlapping predictions

class RetinaFaceDetector(FaceDetector):
    logger = logging.getLogger(__name__)
    
    def __init__(
        self, 
        model_name: str = RetinaFaceWeights.MNET_025,
        confidence_threshold: float = 0.5,
        pre_nms_top_k=5000,
        nms_threshold=0.4,
        post_nms_top_k=750,
        dynamic_size: bool = True,
        input_size: tuple[int,int] = (640, 640)
    ):
        super().__init__(__class__.__name__)
        # https://github.com/yakhyo/uniface/blob/main/uniface/retinaface.py
        self.detector = RetinaFace(
            model_name=model_name,
            conf_thresh=confidence_threshold,
            pre_nms_topk=pre_nms_top_k,
            nms_thresh=nms_threshold,
            post_nms_topk=post_nms_top_k,
            dynamic_size=dynamic_size,
            input_size=input_size,
        )
        self.model_name = model_name
        self.input_size = input_size
    
    def detect_faces(self, image: MatLike) -> list[tuple[int,int,int,int]]:
        scale_factor_w = 1.0
        scale_factor_h = 1.0
        # the reason for dynamic resizing is that RetinaFace works best when the image size is close to the input size
        # if the image is too small, it will miss faces
        # if the image is too large, it will be slow and may not detect faces well
        # so we resize the image to be close to the input size while maintaining the aspect ratio
        if self.detector.dynamic_size:
            h, w, _ = image.shape
            scale_factor_w = self.input_size[0] / w if w > self.input_size[0] else 1.0
            scale_factor_h = self.input_size[1] / h if h > self.input_size[1] else 1.0
            self.logger.info(f"Dynamic resizing enabled. Scale factor: width: {scale_factor_w}, height: {scale_factor_h}")
        detections = self.detector.detect(Preprocessor.resize(image, target_size=self.input_size))
        # Unpack detections
        # see why we convert to xywh format:
        # https://github.com/yakhyo/uniface/blob/main/uniface/visualization.py
        # https://github.com/yakhyo/uniface/blob/main/uniface/retinaface.py
        boxes, landmarks = detections
        scores = boxes[:, 4]
        bboxes = boxes[:, :4]
        # convert from [x1, y1, x2, y2] to [x, y, w, h] and scale back to original image size
        bboxes = [
            Preprocessor.xyxy_to_xywh(bbox) * np.array([
                1 / scale_factor_w, 1 / scale_factor_h, 
                1 / scale_factor_w, 1 / scale_factor_h
            ])
            for bbox in bboxes if bbox is not None
        ]
        landmarks = [
            np.array(landmark) * np.array([
                1 / scale_factor_w, 1 / scale_factor_h
            ]) * (len(landmark) // 2)
            for landmark in landmarks if landmark is not None
        ]

        bboxes = np.array(bboxes).astype(np.int32)
        landmarks = np.array(landmarks).astype(np.int32)

        self.logger.info(f"Detected {len(bboxes)} face(s)")
        self.logger.info(f"""\n
            bboxes: {pprint.pformat(bboxes)},
            landmarks: {pprint.pformat(landmarks)},
            scores: {pprint.pformat(scores)}
        """)
        return bboxes

    def settings(self):
        return {
            "model_name": self.get_name(),
            "model": self.model_name,
            "input_size": self.input_size,
            "onnx_runtime_version": ort.__version__,
            "onnx_meta": self.detector.session.get_modelmeta(),
            "confidence_threshold": self.detector.conf_thresh,
            "pre_nms_top_k": self.detector.pre_nms_topk,
            "nms_threshold": self.detector.nms_thresh,
            "post_nms_top_k": self.detector.post_nms_topk,
            "dynamic_size": self.detector.dynamic_size,
        }