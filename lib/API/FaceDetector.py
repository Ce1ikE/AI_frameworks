# interface for the Pipeline class
# Implementations of this class only need to make sure to output
# bounding boxes aka ROI
# e.g.:
# - ViolaJonesDetector
# - RetinaFaceDetector
# - MTCNNDetector

from PIL.Image import Image
from abc import ABC,abstractmethod

class FaceDetector():
    def __init__(self,):
        pass

    @abstractmethod    
    def detect_faces(self, image: Image) -> list[tuple[int,int,int,int]]:
        """Return a list of bounding boxes [(x, y, w, h), ...]"""
        raise NotImplementedError
    
    @abstractmethod
    def detect_and_draw_faces(self, image: Image) -> tuple[list[tuple[int,int,int,int]], Image]:
        """Return a list of bounding boxes [(x, y, w, h), ...] and draw them on the original input image"""
        raise NotImplementedError