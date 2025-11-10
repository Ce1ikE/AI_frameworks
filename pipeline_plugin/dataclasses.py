from typing import Optional, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import pandas as pd

class WorkerKeys(Enum):
    ANNOTATED_RECORDS = "ANNOTATED_RECORDS" 
    TRAINING_RECORDS = "TRAINING_RECORDS"
    EXTRACTION_RECORDS = "IMAGE_RESULTS_RECORDS"

class PipelineKeys(Enum):
    AGGREGATED_RECORDS = "AGGREGATED_RECORDS" 
    REPORT_RECORDS = "REPORT_RECORDS"

class ExportKeys(Enum):
    IMAGE_NAME = "image_name" 
    FACE_INDEX = "face_index" 
    EMBEDDING = "embedding" 
    EMBEDDING_NORM = "embedding_norm" 
    BBOX = "bbox" 
    LANDMARKS = "landmarks" 
    CONFIDENCE_SCORE = "confidence_score" 
    FACE_IMAGE = "face_image" 
    LABEL = "label"

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

@dataclass
class FileMessage:
    filename: str = field(default=None)  # name of the file without extension
    content: Any = field(default=None)  # raw file data as bytes or string
    file_format: 'Format' = field(default=None)

    class Format(Enum):
        JSON = "json"
        CSV = "csv"
        TXT = "txt"
        BIN = "bin"
        PARQUET = "parquet"
        JPG = "jpg"
        PNG = "png"
        SVG = "svg"
        HTML = "html"
        ONNX = "onnx"
        NPY = "npy"

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# from high level to low level:
# image
#   |-> detections --> subscriber (e.g. annotated images)
#       |-> faces --> subscriber (e.g. cropped faces)
#           |-> embeddings --> subscriber (e.g. embeddings as numpy files)
#               |-> classifications --> subscriber (e.g. classified labels as text files)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 1) ImageMessage
@dataclass
class ImageMessage:
    path: Path = field(default=None)  # optional path to the image file
    image: np.ndarray = field(default=None)  # shape: (height, width, channels)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 2) single face detection 
@dataclass
class FaceDetectionMessage:
    # (x1, y1, x2, y2)
    bbox: BoundingBox = field(default=None)  
    score: float = field(default=0.0)
    # [(x, y), ...] for eyes, nose, mouth corners
    landmarks: Optional[np.ndarray] = field(default=None)  
# 2) image with multiple detections
@dataclass
class ImageDetectionMessage:
    original_image: ImageMessage = field(default=None)
    detections: List[FaceDetectionMessage] = field(default_factory=list)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 3) single face (cropped and aligned)
@dataclass
class FaceMessage:
    # cropped (and aligned) face image
    face_image: ImageMessage = field(default=None)
    detection: FaceDetectionMessage = field(default=None)

# 3) multiple faces in one image
@dataclass
class ImageFaceMessage:
    # processed face image
    faces: List[FaceMessage] = field(default_factory=list)
    original_image: Optional[ImageMessage] = field(default=None)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 4) image with multiple face embeddings
@dataclass
class FaceEmbeddingMessage:
    embedding: np.ndarray = field(default=None)
    face: Optional[FaceMessage] = field(default=None)
# 4) image with multiple face embeddings
@dataclass
class ImageEmbeddingMessage:
    embeddings: List[FaceEmbeddingMessage] = field(default_factory=list)
    original_image: Optional[ImageMessage] = field(default=None)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 5) classified face
@dataclass
class FaceClassifiedMessage:
    label: str = field(default=None)
    embedding: Optional[FaceEmbeddingMessage] = field(default=None)
# 5) image with multiple classified faces
@dataclass
class ImageClassifiedMessage:
    classifications: List[FaceClassifiedMessage] = field(default_factory=list)
    original_image: Optional[ImageMessage] = field(default=None)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
@dataclass
class Embeddings:
    source: Path =  field(default_factory=None)
    embeddings: pd.DataFrame = field(default_factory=pd.DataFrame)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
@dataclass
class TrainedModel:
    model_name: str = field(default_factory=None)
    model: Any =  field(default_factory=None)
    training_time: int = field(default_factory=None)

@dataclass
class TrainingResults:
    embeddings: Embeddings =  field(default_factory=None)
    models: list[TrainedModel] =  field(default_factory=None)
    
      

