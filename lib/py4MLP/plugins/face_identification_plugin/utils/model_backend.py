from enum import Enum
from abc import ABC, abstractmethod


class BackendType(Enum):
    """
        Enum for model backend types
    """
    ONNX = "onnx"
    SKLEARN = "sklearn"
    TORCH = "torch"
    TENSORFLOW = "tensorflow"
    OPENCV = "opencv"
    CUSTOM = "custom"

class BackendMixin(ABC):
    backend_type: BackendType
    device: str = "cpu"

    def to(self, device: str):
        """
            Move the model to the specified device
        """
        if not hasattr(self, 'supports_device'):
            raise NotImplementedError("Backend does not support device management")
        self.device = device
        return self
    
    def is_onnx(self) -> bool:
        return self.backend_type == BackendType.ONNX
    
    def is_sklearn(self) -> bool:
        return self.backend_type == BackendType.SKLEARN
    
    def is_torch(self) -> bool:
        return self.backend_type == BackendType.TORCH
    
    def is_tensorflow(self) -> bool:
        return self.backend_type == BackendType.TENSORFLOW
    
    def is_opencv(self) -> bool:
        return self.backend_type == BackendType.OPENCV
    
    def is_custom(self) -> bool:
        return self.backend_type == BackendType.CUSTOM
    

