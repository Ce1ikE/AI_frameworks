# interface for the Pipeline class
# Implementations of this class only need to make sure to output
# bounding boxes aka ROI
# e.g.:
# - ViolaJonesDetector
# - RetinaFaceDetector
# - MTCNNDetector

from .ModelWrapper import ModelWrapper
from PIL.Image import Image
from abc import ABC,abstractmethod

class FaceDetector(ModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    @abstractmethod    
    def detect_faces(self, image):
        """Return a list of bounding boxes [(x, y, w, h), ...] and optionally landmarks and scores"""
        raise NotImplementedError
    