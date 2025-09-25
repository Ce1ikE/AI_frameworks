from enum import Enum
from ..API.FaceDetector import FaceDetector
from cv2.typing import MatLike 
import logging
from ..API.model_store import verify_model_weights
from typing import Dict

class SCRFDWeights(str, Enum):
    SCRFD_500M = "scrfd_500m"
    SCRFD_1G = "scrfd_1g"
    SCRFD_2G = "scrfd_2g"
    SCRFD_10G = "scrfd_10g"

MODEL_URLS: Dict[SCRFDWeights, str] = {
    SCRFDWeights.SCRFD_500M: 'https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx',
    SCRFDWeights.SCRFD_1G: 'https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_1g.onnx',
    SCRFDWeights.SCRFD_2G: 'https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2g.onnx',
    SCRFDWeights.SCRFD_10G: 'https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx',
}
MODEL_SHA256: Dict[SCRFDWeights, str] = {
    SCRFDWeights.SCRFD_500M: '',
    SCRFDWeights.SCRFD_1G: '',
    SCRFDWeights.SCRFD_2G: '',
    SCRFDWeights.SCRFD_10G: '',
}
CHUNK_SIZE = 8192

# https://github.com/yakhyo/face-reidentification/blob/main/models/scrfd.py
class SCRFDDetector(FaceDetector):
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        model_name: str = None,
    ):
        super().__init__(__class__.__name__ if model_name is None else model_name)

        self.model_path = verify_model_weights(
            model_name=model_name,
            model_urls=MODEL_URLS,
            model_sha256=MODEL_SHA256,
            root="~/.scrfd/weights",
            chunk_size=CHUNK_SIZE
        )

    def detect_faces(self, image: MatLike) -> list[tuple[int, int, int, int]]:
        self.logger.debug("Detecting faces...")
        # Implement face detection logic here
        return []
