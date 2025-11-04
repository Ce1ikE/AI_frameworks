import cv2
import logging
import numpy as np


logger = logging.getLogger(__name__)

class Utils:
    def __init__( target_size=(160, 160)):
        target_size = target_size
    
    @staticmethod
    def resize(image, target_size=(160, 160)):
        return cv2.resize(image, target_size)

    @staticmethod
    def xywh_to_xyxy(bbox):
        x, y, w, h = bbox
        return np.array([x, y, x + w, y + h])

    @staticmethod
    def xyxy_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])

    @staticmethod
    def validate_instance(obj, cls, name: str):
        if not isinstance(obj, cls):
            raise ValueError(f"{name} must be an instance of {cls.__name__}")
    