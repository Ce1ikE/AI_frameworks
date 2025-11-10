from enum import Enum
from typing import Dict
import onnxruntime as ort

from .base import FaceDetector
from ..dataclasses import ImageMessage
from ..utils.model_store import verify_model_weights

class SCRFDWeights(str, Enum):
    SCRFD_500M = "scrfd_500m"
    SCRFD_1G = "scrfd_1g"
    SCRFD_2G = "scrfd_2g"
    SCRFD_10G = "scrfd_10g"
    SCRFD_34G = "scrdf_34g"

MODEL_URLS: Dict[SCRFDWeights, str] = {
    SCRFDWeights.SCRFD_500M: 'https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx',
    SCRFDWeights.SCRFD_1G: 'https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_1g.onnx',
    SCRFDWeights.SCRFD_2G: 'https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2g.onnx',
    SCRFDWeights.SCRFD_10G: 'https://huggingface.co/kunkunlin1221/face-detection_scrfd-10g-gnkps/raw/main/scrfd_10g_gnkps_fp32.onnx',
    SCRFDWeights.SCRFD_34G: 'https://huggingface.co/immich-app/scrfd_34g_gnkps/raw/main/detection/model.onnx',
}
MODEL_SHA256: Dict[SCRFDWeights, str] = {
    SCRFDWeights.SCRFD_500M: '',
    SCRFDWeights.SCRFD_1G: '',
    SCRFDWeights.SCRFD_2G: '',
    SCRFDWeights.SCRFD_10G: '2112d066c1dce6cc648670e69cf90561b9287bb1945153f3b461a487131255b9',
    SCRFDWeights.SCRFD_34G: 'aa19f0e7f4d120d4cf990086639ab74a0136adceaebd232e0dc4745e0cfd4257',
}
CHUNK_SIZE = 8192

# https://github.com/yakhyo/face-reidentification/blob/main/models/scrfd.py
class SCRFDDetector(FaceDetector):
    def __init__(
        self, 
        model_dir: str, 
        model_name: SCRFDWeights = SCRFDWeights.SCRFD_2G,
        device: str = "cpu",
    ):
        super().__init__()

        self.model_path = verify_model_weights(
            model_name=model_name,
            model_urls=MODEL_URLS,
            model_sha256=MODEL_SHA256,
            root=model_dir / "arcface",
            chunk_size=CHUNK_SIZE
        )
        self.device = device
        self.model_name = model_name
        self.initialized = False
        self.session = None

    def _lazy_init(self):
        if not self.initialized:
            providers = ["CPUExecutionProvider"] if self.device == "cpu" else ["CUDAExecutionProvider"]

            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.initialized = True

    def detect_faces(self, image: ImageMessage):
        self._lazy_init()
        return []
    
    def settings(self):
        return {
            **super().settings(),
            "model_name": self.model_name,
        }
