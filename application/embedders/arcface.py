import cv2
import logging
import numpy as np
from enum import Enum
from typing import Dict
from pathlib import Path
import onnxruntime as ort


from .base import FaceEmbedder
from ..utils.model_backend import BackendType
from ..dataclasses import FaceEmbeddingMessage, FaceMessage
from ..utils.model_store import verify_model_weights

class ArcFaceWeights(str, Enum):
    W600K_MBF = "arcface_w600k_mbf"
    """
        model based on MobileFaceNet architecture.
        W600K (webface 600,000 identities) dataset used for training
    """
    W600K_R50 = "arcface_w600k_r50"
    """
        model based on ResNet50 architecture. 
        W600K (webface 600,000 identities) dataset used for training
    """

MODEL_URLS: Dict[ArcFaceWeights, str] = {
    ArcFaceWeights.W600K_MBF: 'https://huggingface.co/WePrompt/buffalo_sc/resolve/main/w600k_mbf.onnx?download=true',
    ArcFaceWeights.W600K_R50: 'https://huggingface.co/maze/faceX/resolve/main/w600k_r50.onnx?download=true',
}

MODEL_SHA256: Dict[ArcFaceWeights, str] = {
    ArcFaceWeights.W600K_MBF: '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f',
    ArcFaceWeights.W600K_R50: '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43',
}

CHUNK_SIZE = 8192

# https://github.com/yakhyo/face-reidentification/blob/main/models/arcface.py
class ArcFaceEmbedder(FaceEmbedder):
    def __init__(
        self, 
        model_dir: str, 
        model_name: ArcFaceWeights = ArcFaceWeights.W600K_MBF,
        device: str = "cpu",
    ):
        super().__init__(
            backend=BackendType.ONNX
        )
        # download model from URL
        self.model_path = verify_model_weights(
            model_name=model_name,
            root=model_dir / "arcface",
            model_urls=MODEL_URLS,
            model_sha256=MODEL_SHA256,
            chunk_size=CHUNK_SIZE
        )
        self.model_name = model_name
        self.device = device

        self.input_shape = (112, 112)
        self.output_shape = (512,)

        self.session = None
        self.initialized = False

    def _lazy_init(self):
        if not self.initialized:
            # either use onnxruntime's preload_dlls() method (newer versions)
            # or import torch. 
            if self.device != "cpu":
                ort.preload_dlls()
        
            providers = ["CPUExecutionProvider"] if self.device == "cpu" else ["CUDAExecutionProvider"]
            self.session = ort.InferenceSession(self.model_path,  providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.initialized = True

    # color space converted to RGB 
    # dimensions changed from HWC -> CHW (C = Channel, H = Height, W = Width)
    # resized to 112x112
    # normalized to [-1, 1]
    # batch dimension added
    # returns a shape of: (1, 3, 112, 112) (for ArcFace)
    def preprocess(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, self.input_shape)
        face_img = (face_img.astype(np.float32) - 127.5) / 128.0
        face = np.transpose(face_img, (2, 0, 1))[np.newaxis,...]
        return face 
    
    def postprocess(self, embedding: np.ndarray):
        return (embedding / np.linalg.norm(embedding)).flatten()

    def embed_face(self, message: FaceMessage) -> np.ndarray:
        self._lazy_init()
        face_img = self.preprocess(message.face_image.image)
        embedding = self.session.run(None, {self.input_name: face_img})[0]
        embedding = self.postprocess(embedding)
        return embedding
    
    def settings(self):
        return {
            **super().settings(),
            "model": self.model_name.value,
            "device": self.device,
            "input_size": self.input_shape,
            "embedding_size": self.output_shape,
            "onnx_runtime_version": ort.__version__,
        }