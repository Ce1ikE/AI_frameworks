from cv2.typing import MatLike 
import logging
import numpy as np
import pprint
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
# 4) NMS removes overlapping predictions
class RetinaFaceDetector(FaceDetector):
    logger = logging.getLogger(__name__)
    
    def __init__(
        self, 
        model_path=RetinaFaceWeights.MNET_025,
        model_name: str = None,
        confidence_threshold: float = 0.5,
    ):
        super().__init__(__class__.__name__ if model_name is None else model_name)
        # https://github.com/yakhyo/uniface/blob/main/uniface/retinaface.py
        self.confidence_threshold = confidence_threshold
        self.detector = RetinaFace(
            model_name=model_path,
            conf_thresh=confidence_threshold,
            pre_nms_topk=5000,
            nms_thresh=0.4,
            post_nms_topk=750,
            dynamic_size=True,
        )

    
    def detect_faces(self, image: MatLike) -> list[tuple[int,int,int,int]]:
        detections = self.detector.detect(image)
        # Unpack detections
        # see why we convert to xywh format:
        # https://github.com/yakhyo/uniface/blob/main/uniface/visualization.py
        boxes, landmarks = detections
        scores = boxes[:, 4]
        bboxes = boxes[:, :4].astype(np.int32).tolist()
        # convert from [x1, y1, x2, y2] to [x, y, w, h]        
        bboxes = [Preprocessor.xyxy_to_xywh(bbox) for bbox in bboxes if bbox is not None]
        
        self.logger.info(f"Detected {len(bboxes)} face(s)")
        self.logger.info(f"""\n
            bboxes: {pprint.pformat(bboxes)},
            landmarks: {pprint.pformat(landmarks)},
            scores: {pprint.pformat(scores)}
        """)
        return bboxes

 