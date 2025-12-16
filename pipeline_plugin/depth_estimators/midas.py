from email import message
import numpy as np
from pathlib import Path
from enum import Enum
from typing import Dict, Literal
import onnxruntime as ort
import cv2

from ..dataclasses import ImageFaceMessage
from ..utils.model_store import verify_model_weights

class MiDaSWeights(str, Enum):
    MIDAS_V21_SMALL_256 = "midas_v21_small_256"
    """
        MiDaS v2.1 Small model with input size 256x256
        optimized for speed and lower memory consumption
    """
    DPT_LARGE_384 = "dpt_large_384"
    """
        DPT Large model with input size 384x384
        optimized for accuracy over speed
    """
    DPT_SWIN2_TINY_256 = "dpt_swin2_tiny_256"
    """
        DPT Swin2 Tiny model with input size 256x256
        optimized for speed with reasonable accuracy
    """


MODEL_URLS: Dict[MiDaSWeights, str] = {
    MiDaSWeights.MIDAS_V21_SMALL_256: 'https://huggingface.co/julienkay/sentis-MiDaS/resolve/main/onnx/midas_v21_small_256.onnx',
    MiDaSWeights.DPT_LARGE_384: 'https://huggingface.co/julienkay/sentis-MiDaS/resolve/main/onnx/dpt_large_384.onnx',
    MiDaSWeights.DPT_SWIN2_TINY_256: 'https://huggingface.co/julienkay/sentis-MiDaS/resolve/main/onnx/dpt_swin2_tiny_256.onnx',
}

MODEL_SHA256: Dict[MiDaSWeights, str] = {
    MiDaSWeights.MIDAS_V21_SMALL_256: 'b0a5b3f12625137e626805167907fe0410665bec671685d59daaa2daab19f977',
    MiDaSWeights.DPT_LARGE_384: '42b2e08dada8bd0e4612ac268f5f92c065389484c7e4cfab4ca8f3c32a13090f',
    MiDaSWeights.DPT_SWIN2_TINY_256: '9590c809dbd1930b020762f22160e75eae388bd8cc65b779c47fdd689618d804'
}

CHUNK_SIZE = 8192

class MiDaSEstimator:
    def __init__(
        self,
        model_dir: Path,
        model_name: MiDaSWeights = MiDaSWeights.MIDAS_V21_SMALL_256,
        device: Literal["cpu", "cuda"] = "cpu",
        depth_threshold: float = 0.9,
    ):
        self.model_path = verify_model_weights(
            model_name,
            model_dir,
            MODEL_URLS,
            MODEL_SHA256
        )
        self.model_name = model_name.value 
        self.device = device
        self.session: ort.InferenceSession | None = None
        self.initialized = False
        self.threshold = depth_threshold

    def _lazy_init(self):
        if not self.initialized:
            if self.device != "cpu":
                ort.preload_dlls()
            providers = ["CPUExecutionProvider"] if self.device == "cpu" else ["CUDAExecutionProvider"]
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.initialized = True

    def foreground_mask_from_depth(
        self,   
        depth_map: np.ndarray,
        keep_percentile: float = 90.0
    ) -> np.ndarray:
        """
        keeps the closest `keep_percentile` depth values.
        depth_map must be normalized to [0, 1].
        """
        thresh = np.percentile(depth_map, keep_percentile)
        mask = depth_map >= thresh
        return mask.astype(np.uint8)
    
    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def bounding_box_from_mask(self, mask: np.ndarray):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # keep largest connected foreground region
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return x, y, x + w, y + h
    
    def cut_image(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Crops image to closest foreground region using depth.
        """
        mask = self.foreground_mask_from_depth(depth_map, keep_percentile=self.threshold * 100)
        mask = self.clean_mask(mask)

        bbox = self.bounding_box_from_mask(mask)
        if bbox is None:
            return image

        x1, y1, x2, y2 = bbox

        # optional padding to avoid cutting faces too tight
        pad = int(0.05 * max(image.shape[:2]))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)
        return image[y1:y2, x1:x2]


    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        self._lazy_init()
        input_image = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_image})
        depth = outputs[0]

        if depth.ndim == 4:
            depth_map = depth[0, 0]
        elif depth.ndim == 3:
            depth_map = depth[0]
        else:
            raise RuntimeError(f"Unexpected depth output shape: {depth.shape}")

        return depth_map
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2
        input_size = self.model_name.split("_")[-1]
        input_size = int(input_size)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img
    
    def remove_background(
        self,
        image: np.ndarray,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Removes background pixels based on depth.
        Returns same-sized image.
        """
        mask = self.foreground_mask_from_depth(
            depth_map,
            keep_percentile=self.threshold * 100
        )

        mask = self.clean_mask(mask)

        # ensure mask shape (H, W, 1)
        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        # apply mask
        foreground = image.astype(np.float32) * mask

        return foreground.astype(image.dtype)
    
    def normalize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        min_val = np.min(depth_map)
        max_val = np.max(depth_map)
        normalized_depth = (depth_map - min_val) / (max_val - min_val + 1e-8)
        return normalized_depth
    
    def resize_depth_map(self, depth_map: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        resized_depth = cv2.resize(depth_map, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
        return resized_depth
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        depth_map = self.estimate_depth(image)
        normalized_depth = self.normalize_depth_map(depth_map)
        resized_depth = self.resize_depth_map(normalized_depth, image.shape[:2])
        foreground = self.remove_background(image, resized_depth)
        return foreground
    
    def settings(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "device": self.device
        }

