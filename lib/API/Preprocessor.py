import cv2
from pathlib import Path
import numpy as np
class Preprocessor:
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size

    @staticmethod
    def load(image_path: Path):
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return cv2.imread(image_path.as_posix())

    @staticmethod
    def crop(image, bbox):
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def scale(image, scale_factor=1.0):
        if scale_factor <= 0:
            raise ValueError("Scale factor must be positive.")
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        return cv2.resize(image, (width, height))
    
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
