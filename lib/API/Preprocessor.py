import cv2
from pathlib import Path

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
    def resize(image, target_size=(160, 160)):
        return cv2.resize(image, target_size)

    @staticmethod
    def xwyh_to_xywh(bbox):
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    @staticmethod
    def xyxy_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
