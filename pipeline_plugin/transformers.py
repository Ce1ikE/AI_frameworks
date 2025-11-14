from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from sklearn.base import BaseEstimator, ClusterMixin
from typing import Callable

from lib.py4MLP.core.component import *
from lib.py4MLP.core.pipeline import *
from lib.py4MLP.core.bus import *
from .utils.util_classes import *

from .transformers import *
from .dataclasses import *

from .detectors import *
from .embedders import *
from .classifiers import *

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
#  image -> face detection -> face embedding (extraction)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ImageFileLoader(Source):
    """
        Source element that loads images from a path
        and publishes image data to the pipeline
    """
    def __init__(self, name: str,data: Path | list[Path]):
        super().__init__(name)
        self.output_type = ImageMessage
        self.data = data 

    def process(self):
        for image_path in self.data:
            image_path: Path = Path(image_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            suffix = image_path.suffix.lower()

            if suffix in [".jpg", ".jpeg", ".png"]:
                image = cv2.imread(str(image_path))
            elif suffix == ".heic":
                try:
                    from pillow_heif import register_heif_opener
                    from PIL import Image
                except ImportError as e:
                    raise ImportError("pillow_heif required for HEIC images") from e

                register_heif_opener()
                with Image.open(image_path) as img:
                    image = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Unsupported image format: {suffix}")

            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            yield ImageMessage(path=image_path, image=image)

    def settings(self):
        return {
            "data": str(self.data.relative_to(Path.cwd())) 
            if isinstance(self.data, Path) 
            else [str(p.relative_to(Path.cwd())) for p in self.data]
        }

class ImageFacesDetector(Transformer):
    """
        Transformer element that detects faces in an image
    """
    def __init__(self, name: str, detector: FaceDetector):
        super().__init__(name)
        Utils.validate_instance(detector, FaceDetector, "detector")
        self.detector = detector

    def process(self, data: ImageMessage):
        try:
            bboxes, scores, landmarks = self.detector.detect_faces(data)
        except Exception as e:
            raise RuntimeError(f"Failed to detect faces {e}")

        return ImageDetectionMessage(
            original_image=data,
            detections=[
                FaceDetectionMessage(
                    bbox=BoundingBox(
                        x1=int(bbox[0]),
                        y1=int(bbox[1]),
                        x2=int(bbox[2]),
                        y2=int(bbox[3])
                    ),
                    landmarks=landmarks[i] if landmarks is not None and len(landmarks) > i else None,
                    score=scores[i] if scores is not None and len(scores) > i else None
                )
                for i, bbox in enumerate(bboxes)
            ]
        )
    
    def settings(self):
        return {
            "detector": self.detector.__class__.__name__,
            "model": self.detector.settings()
        }

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ImageFacesExtractor(Transformer):
    """
        Transformer element that extracts faces from an image
        by cropping and optionally aligning them based on detections
    """
    def __init__(self, name: str,min_face_ratio: int = 0.02):
        super().__init__(name)
        self.min_face_ratio = min_face_ratio  

    def extract_face(self, image: np.ndarray, detection: FaceDetectionMessage) -> np.ndarray:
        if detection.landmarks is not None and len(detection.landmarks) == 5:
            from uniface import face_alignment
            face, _ = face_alignment(image, detection.landmarks)
        else:
            b = detection.bbox
            face = image[b.y1:b.y2, b.x1:b.x2]
        return face

    def process(self, data: ImageDetectionMessage):
        faces = []

        for det in data.detections:
            try:
                face_image = self.extract_face(
                    data.original_image.image, 
                    det
                )
                faces.append(
                    FaceMessage(
                        face_image=ImageMessage(
                            path=None, 
                            image=face_image
                        ),
                        detection=det
                    )
                )
            except Exception as e:
                raise RuntimeError(f"Failed to extract face: {e}")
            
        return ImageFaceMessage(
            faces=faces, 
            original_image=data.original_image
        )
    
    def settings(self):
        return {
            "extractor": self.__class__.__name__
        }
    
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ImageFacesEmbedder(Transformer):
    """
        Transformer element that embeds faces using a face embedder
    """
    def __init__(self, name: str, embedder: FaceEmbedder):
        super().__init__(name)
        Utils.validate_instance(embedder, FaceEmbedder, "embedder")
        self.embedder = embedder

    def process(self, data: ImageFaceMessage):
        return ImageEmbeddingMessage(
            original_image=data.original_image,
            embeddings=[
                FaceEmbeddingMessage(
                    embedding=self.embedder.embed_face(face),
                    face=face
                )
                for face in data.faces
            ] 
        )
        
    def settings(self):
        return {
            "embedder": self.embedder.__class__.__name__,
            "model": self.embedder.settings()
        }

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Transformers for embeddings -> training a classifier 
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EmbeddingFileLoader(Source):
    def __init__(self, name: str, data: Path | list[Path]):
        super().__init__(name)
        self.data = data

    def process(self):
        for data in (self.data if isinstance(self.data, list) else [self.data]):
            data: Path = Path(data)
            if not data.is_file():
                raise FileNotFoundError(f"Embedding file not found: {data}")
        
        for data in (self.data if isinstance(self.data, list) else [self.data]):
            data: Path = Path(data)
            try:
                df = pd.read_parquet(data)
            except Exception as e:
                raise ValueError(f"Failed to load embeddings from {data}: {e}")

            yield Embeddings(source=data, embeddings=df)

    def settings(self):
        return {
            "data": str(self.data.relative_to(Path.cwd())) 
            if isinstance(self.data, Path) 
            else [str(p.relative_to(Path.cwd())) for p in self.data]
        }

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EmbeddingTrainer(Transformer):
    def __init__(self, name: str, algorithms: list[BaseEstimator | ClusterMixin]):
        super().__init__(name)
        self.algorithms = algorithms

    def process(self, data: Embeddings):
        try:
            df = data.embeddings
        except Exception as e:
            raise ValueError("Embeddings not found for training") from e
        if df is None or len(df) == 0:
            raise ValueError("No embeddings available for training")

        key = ExportKeys.EMBEDDING_NORMALIZED.value
        if key not in df.columns:
            print("Normalizing embeddings...")
            # (N, D) matrix
            raw_embeddings = np.vstack(df[ExportKeys.EMBEDDING.value].values)
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10  # avoid divide by zero
            normalized = raw_embeddings / norms
            df[key] = list(normalized)
            df[ExportKeys.EMBEDDING.value] = list(normalized)
            embeddings = normalized
        else:
            print("Found normalized embeddings in dataframe.")
            embeddings = np.vstack(df[key].values)
        

        training_times = []
        for alg in self.algorithms:
            print(f'Training {alg.__class__.__name__} ...')
            start_time =  dt.datetime.now()
            labels = alg.fit_predict(embeddings)
            end_time = dt.datetime.now()
            df[alg.__class__.__name__] = labels
            time_diff = end_time - start_time
            training_times.append(time_diff.total_seconds())
            print(f'Training finished for {alg.__class__.__name__}')

        return TrainingResults(
            embeddings=Embeddings(
                source=data.source,
                embeddings=df,
            ),
            models=[
                TrainedModel(
                    model=alg,
                    model_name=alg.__class__.__name__,
                    training_time=train_time_alg
                ) 
                for train_time_alg, alg in zip(training_times,self.algorithms)
            ]
        )

    def settings(self):
        return {
            "trained_algorithms" : [{algorithm.__class__.__name__ : algorithm.get_params()} for algorithm in self.algorithms],
        }

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EmbeddingClassifier(Transformer):
    def __init__(self, name, classifier: FaceClassifier):
        super().__init__(name)
        Utils.validate_instance(classifier, FaceClassifier, "classifier")
        self.classifier = classifier

    def process(self, data: ImageEmbeddingMessage):
        
        classifications: list[FaceClassifiedMessage] = []
        if data.embeddings is not None:
            for embedding_message in data.embeddings:
                emb_normalized = embedding_message.embedding / np.linalg.norm(embedding_message.embedding)
                classifications.append(
                    FaceClassifiedMessage(
                        label=self.classifier.predict(emb_normalized),
                        embedding=embedding_message
                    )
                )

        return ImageClassifiedMessage(
            original_image=data.original_image,
            classifications=classifications
        )
    
    def settings(self):
        return {
            "classifier" : self.classifier.settings()
        }