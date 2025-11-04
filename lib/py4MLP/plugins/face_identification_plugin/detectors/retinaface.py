
import numpy as np
from enum import Enum
from typing import Dict, Literal
import onnxruntime as ort

from .base import FaceDetector
from ..utils.model_backend import BackendType
from ..utils.model_store import verify_model_weights
from ..dataclasses import ImageMessage
from uniface.common import (
    nms,
    resize_image,
    decode_boxes,
    generate_anchors,
    decode_landmarks
)

class RetinaFaceWeights(str, Enum):
    MNET_025 = "retinaface_mnet025"
    """model based on MobileNetV1 architecture with width multiplier 0.25"""
    MNET_050 = "retinaface_mnet050"
    """model based on MobileNetV1 architecture with width multiplier 0.50"""
    MNET_V1  = "retinaface_mnet_v1"
    """model based on MobileNetV1 architecture"""
    MNET_V2  = "retinaface_mnet_v2"
    """model based on MobileNetV2 architecture"""
    RESNET18 = "retinaface_r18"
    """model based on ResNet18 architecture"""
    RESNET34 = "retinaface_r34"
    """model based on ResNet34 architecture"""


MODEL_URLS: Dict[RetinaFaceWeights, str] = {
    RetinaFaceWeights.MNET_025: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1_0.25.onnx',
    RetinaFaceWeights.MNET_050: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1_0.50.onnx',
    RetinaFaceWeights.MNET_V1:  'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1.onnx',
    RetinaFaceWeights.MNET_V2:  'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv2.onnx',
    RetinaFaceWeights.RESNET18: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_r18.onnx',
    RetinaFaceWeights.RESNET34: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_r34.onnx'
}

MODEL_SHA256: Dict[RetinaFaceWeights, str] = {
    RetinaFaceWeights.MNET_025: 'b7a7acab55e104dce6f32cdfff929bd83946da5cd869b9e2e9bdffafd1b7e4a5',
    RetinaFaceWeights.MNET_050: 'd8977186f6037999af5b4113d42ba77a84a6ab0c996b17c713cc3d53b88bfc37',
    RetinaFaceWeights.MNET_V1:  '75c961aaf0aff03d13c074e9ec656e5510e174454dd4964a161aab4fe5f04153',
    RetinaFaceWeights.MNET_V2:  '3ca44c045651cabeed1193a1fae8946ad1f3a55da8fa74b341feab5a8319f757',
    RetinaFaceWeights.RESNET18: 'e8b5ddd7d2c3c8f7c942f9f10cec09d8e319f78f09725d3f709631de34fb649d',
    RetinaFaceWeights.RESNET34: 'bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630'
}

CHUNK_SIZE = 8192

# https://medium.com/axinc-ai/retinaface-a-face-detection-model-designed-for-high-resolution-6c3900771a01
class RetinaFaceDetector(FaceDetector):
    
    def __init__(
        self,
        model_dir: str, 
        model_name: RetinaFaceWeights = RetinaFaceWeights.MNET_025,
        device: str = "cpu",
        target_size: tuple[int, int] = (640,640),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        pre_nms_topk: int = 5000,
        post_nms_topk: int = 750,
        decode: bool = False,  
        max_num: int = 0,
        score_metric: Literal["default", "max"] = "default",
        center_weight: float = 2.0
    ):
        super().__init__(
            backend=BackendType.ONNX
        )
        self.model_path = verify_model_weights(
            model_name=model_name,
            root=model_dir / "retinaface",
            model_urls=MODEL_URLS,
            model_sha256=MODEL_SHA256,
            chunk_size=CHUNK_SIZE
        )
        self.model_name = model_name
        self.device = device
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.decode = decode
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.max_num = max_num
        self.score_metric = score_metric
        self.center_weight = center_weight

        self._priors = None 
        self.session = None
        self.initialized = False

    def _lazy_init(self):
        if not self.initialized:
            if self.device != "cpu":
                ort.preload_dlls()
            providers = ["CPUExecutionProvider"] if self.device == "cpu" else ["CUDAExecutionProvider"]
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.initialized = True
            if self.target_size is not None:
                self._priors = generate_anchors(image_size=self.target_size)

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        # scaled to input size
        # mean RGB values ([104, 117, 123]) is substracted
        # dimensions converted from HWC -> CHW
        # add batch dimension (1, C, H, W) 
        if self.target_size is None:
            height, width, _ = image.shape
            # generate anchors for each input image
            self._priors = generate_anchors(image_size=(height, width))  
            # No resizing
            scale = 1.0  
        else:
            image, scale = resize_image(image, target_shape=self.target_size)
        img = np.float32(image) - np.array([104, 117, 123], dtype=np.float32)
        # (1, 3, H, W)
        img = img.transpose(2, 0, 1)[np.newaxis, ...] 
        return img, scale

    def postprocess(self, outputs: list[np.ndarray], scale: float, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray,np.ndarray]:
        loc, conf, landmarks = outputs[0].squeeze(0), outputs[1].squeeze(0), outputs[2].squeeze(0)
        # Decode boxes and landmarks
        boxes = decode_boxes(loc, self._priors)
        landmarks = decode_landmarks(landmarks, self._priors)
       
        _, _, height, width = shape
        bbox_scale = np.array([height, width] * 2)
        boxes = boxes * bbox_scale / scale

        landmark_scale = np.array([height, width] * 5)
        landmarks = landmarks * landmark_scale / scale
        # Extract confidence scores for the face class
        scores = conf[:, 1]
        mask = scores > self.confidence_threshold
        # Filter by confidence threshold
        boxes, landmarks, scores = boxes[mask], landmarks[mask], scores[mask]
        # Sort by scores
        order = scores.argsort()[::-1][:self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]
        # Apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, self.nms_threshold)
        detections, landmarks = detections[keep], landmarks[keep]
        # Keep top-k detections
        detections, landmarks = detections[:self.post_nms_topk], landmarks[:self.post_nms_topk]

        landmarks = landmarks.reshape(-1, 5, 2).astype(np.int32)

        if self.max_num > 0 and detections.shape[0] > self.max_num:
            # Calculate area of detections
            areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])

            # Calculate offsets from image center
            center = (height // 2, width // 2)
            offsets = np.vstack([
                (detections[:, 0] + detections[:, 2]) / 2 - center[1],
                (detections[:, 1] + detections[:, 3]) / 2 - center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)

            # Calculate scores based on the chosen metric
            if self.score_metric == 'max':
                scores = areas
            else:
                scores = areas - offset_dist_squared * self.center_weight

            # Sort by scores and select top `max_num`
            sorted_indices = np.argsort(scores)[::-1][:self.max_num]

            detections = detections[sorted_indices]
            landmarks = landmarks[sorted_indices]
        # https://github.com/yakhyo/uniface/blob/main/uniface/visualization.py
        scores = detections[:, 4]
        bboxes = detections[:, :4]
        return bboxes, scores, landmarks

    def detect_faces(self, message: ImageMessage) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._lazy_init()
        img_input , scale = self.preprocess(message.image)
        outputs = self.session.run(None, {self.input_name: img_input})

        bboxes, scores, landmarks = self.postprocess(outputs, scale,shape=img_input.shape)
        print(f"Detected : {len(bboxes)} face(s)")
        return bboxes, scores, landmarks

    def settings(self):
        return {
            **super().settings(),
            "model": self.model_name.value,
            "device": self.device,
            "target_size": self.target_size,
            "backend": self.backend.name,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "onnx_runtime_version": getattr(ort, "__version__", "N/A"),
        }



    


