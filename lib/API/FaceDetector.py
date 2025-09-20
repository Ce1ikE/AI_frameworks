# interface for the Pipeline class
# Implementations of this class only need to make sure to output
# bounding boxes aka ROI
# e.g.:
# - ViolaJonesDetector
# - RetinaFaceDetector
# - MTCNNDetector

from abc import ABC,abstractmethod

class FaceDetector:
    @abstractmethod    
    def detect_faces(self, image) -> list[tuple[int,int,int,int]]:
        """Return a list of bounding boxes [(x, y, w, h), ...]"""
        raise NotImplementedError