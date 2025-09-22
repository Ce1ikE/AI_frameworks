import cv2
import numpy as np
import  onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
from ..API.FaceEmbedder import FaceEmbedder
import logging

# https://github.com/yakhyo/face-reidentification/blob/main/models/arcface.py
class ArcFaceEmbedder(FaceEmbedder):
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        model_path="arcface.onnx", 
        model_name: str =None
    ):
        super().__init__(__class__.__name__ if model_name is None else model_name)
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    # color space converted to RGB, resized to 112x112, normalized to [-1, 1]
    def preprocess(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (112, 112))
        face_img = (face_img.astype(np.float32) - 127.5) / 128.0
        # https://realpython.com/python-ellipsis/
        # face = np.transpose(face, (2, 0, 1))  # Channel Height, Width
        # shape: (1, 112, 112, 3) (for ArcFace)
        # Batch dimension added
        return face_img[np.newaxis, ...]  

    def embed_face(self, face_img):
        face_img = self.preprocess(face_img)
        try:
            embedding = self.session.run(None, {self.input_name: np.transpose(face_img, (0, 3, 1, 2))})[0]
        except:
            embedding = self.session.run(None, {self.input_name: face_img})[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding  # 512-D vector