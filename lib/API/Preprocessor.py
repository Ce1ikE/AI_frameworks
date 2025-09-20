import cv2
from pathlib import Path

class Preprocessor:
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size

    def load(self, image_path: Path):
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return cv2.imread(image_path.as_posix())
    
    def crop(self, image, bbox):
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    
    def resize(self, image):
        return cv2.resize(image, self.target_size)