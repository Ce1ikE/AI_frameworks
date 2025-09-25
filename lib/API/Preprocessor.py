import cv2
from pathlib import Path
import numpy as np
from PIL.Image import Image
from PIL import Image, UnidentifiedImageError
import logging
from sklearn.base import BaseEstimator
logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size

    @staticmethod
    def load(image_path: Path):
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return cv2.imread(image_path.as_posix())
    
    # https://onnx.ai/sklearn-onnx/index.html
    @staticmethod
    def save_model(model: BaseEstimator, model_path: str, X: np.ndarray):
        from skl2onnx import to_onnx
        onx = to_onnx(model, X[:1].astype(np.float32), target_opset=12)
        with open(model_path, "wb") as f:
            f.write(onx.SerializeToString())

    @staticmethod
    def load_model(model_path: str):
        from onnxruntime import InferenceSession
        with open(model_path, "rb") as f:
            onx = f.read()
        return InferenceSession(onx, providers=["CPUExecutionProvider"])

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

    @staticmethod
    def convert_heic_to_jpg(heic_path: Path, jpg_path: Path):
        # make sure to install pillow-heif for HEIC support
        # either via pip , conda or uv (requires libheif)
        # or use a precompiled binary wheel from:
        # https://pypi.org/project/pillow-heif/
        from pillow_heif import register_heif_opener
        register_heif_opener()
        try:
            with Image.open(heic_path.as_posix()) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(jpg_path.as_posix(), "JPEG")
                logger.info(f"Converted {heic_path} to {jpg_path}")
        except UnidentifiedImageError:
            raise ValueError(f"Cannot identify image file: {heic_path}")
        except Exception as e:
            raise RuntimeError(f"Error converting {heic_path} to JPEG: {e}")