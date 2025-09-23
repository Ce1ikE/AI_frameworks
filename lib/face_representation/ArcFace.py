import cv2
import numpy as np
import onnxruntime as ort
from ..API.model_store import verify_model_weights
from ..API.FaceEmbedder import FaceEmbedder
import logging

from enum import Enum
from typing import Dict

# this code is adapted from:
# https://github.com/yakhyo
# who created the uniface library for RetinaFace
class ArcFaceWeights(str, Enum):
    W600K_MBF = "arcface_w600k_mbf"
    W600K_R50 = "arcface_w600k_r50"

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
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        model_name: str = ArcFaceWeights.W600K_MBF, 
    ):
        super().__init__(__class__.__name__)
        # download model from URL
        self.model_path = verify_model_weights(
            model_name=model_name,
            root="~/.arcface/weights",
            model_urls=MODEL_URLS,
            model_sha256=MODEL_SHA256,
            chunk_size=CHUNK_SIZE
        )
        self.model_name = model_name
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name

    # color space converted to RGB, resized to 112x112, normalized to [-1, 1]
    def preprocess(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (112, 112))
        face_img = (face_img.astype(np.float32) - 127.5) / 128.0
        # https://realpython.com/python-ellipsis/
        # Channel Height, Width
        face = np.transpose(face_img, (2, 0, 1))  
        # shape: (1, 3, 112, 112) (for ArcFace)
        # Batch dimension added
        return face[np.newaxis, ...]  

    def embed_face(self, face_img):
        face_img = self.preprocess(face_img)
        embedding = self.session.run(None, {self.input_name: face_img})[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding  # 512-D vector
    
    def settings(self):
        return {
            "model_name": self.get_name(),
            "model": self.model_name,
            "onnx_runtime_version": ort.__version__,
            "onnx_meta": self.session.get_modelmeta(),
            "input_size": (112, 112),
            "embedding_size": 512,
            "preprocessing": "BGR->RGB, resize to 112x112, normalize to [-1, 1]",
        }