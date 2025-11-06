from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from sklearn.base import BaseEstimator, ClusterMixin

from ...core.component import *
from ...core.pipeline import *
from ...core.bus import *
from .utils.util_functions import *

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
        data_result = []
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

            data_result.append(ImageMessage(path=image_path, image=image))
        return data_result

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
    def __init__(self, name: str):
        super().__init__(name)

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
        if not data.faces:
            return

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
                if "sample_id" not in df.columns or "embedding" not in df.columns or "face_path" not in df.columns:
                    raise ValueError(f"Invalid parquet format in {data}, expected columns: ['sample_id', 'embedding', 'face_path']")

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
class EmbeddingNormalizer(Transformer):
    def __init__(self, name: str):
        super().__init__(name)

    def process(self, data: Embeddings):
        try:
            embeddings = data.embeddings
        except Exception as e:
            raise ValueError("Embeddings not found for normalization") from e

        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings available for normalization")

        embeddings = embeddings['embedding'].tolist()
        X = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1)
        print(f"Average norm before normalization: {norms.mean():.3f}")
        
        if norms.mean() == 1:
            return NormalizedEmbeddings(
                source=data.source,    
                embeddings=data.embeddings
            )
        
        from sklearn.preprocessing import Normalizer
        normalizer = Normalizer(norm='l2')
        norm_embeddings = normalizer.fit_transform(X)
        data.embeddings['embedding'] = pd.Series(norm_embeddings)
        
        return NormalizedEmbeddings(
            source=data.source,    
            embeddings=norm_embeddings
        )

    def settings(self):
        return {
            "normalizer": self.__class__.__name__,
            "norm": "l2"
        }

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EmbeddingTrainer(Transformer):
    def __init__(self, name: str, algorithms: list[BaseEstimator | ClusterMixin],reduce_to=-1):
        super().__init__(name)
        self.algorithms = algorithms
        self.reduce_to = reduce_to

    def process(self, data: NormalizedEmbeddings):
        try:
            embeddings = data.embeddings
        except Exception as e:
            raise ValueError("Embeddings not found for training") from e
        
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings available for training")

        df = data.embeddings
        embeddings = np.vstack(df["embedding"].values)
        if self.reduce_to >= 2:
            from umap import UMAP
            reducer = UMAP(random_state=42,n_components=self.reduce_to)
            reduced_embeddings = reducer.fit_transform(embeddings)
            df["embedding"] = [reduced_embeddings[i, :] for i in range(reduced_embeddings.shape[0])]
        embeddings = np.vstack(df["embedding"].values)

        training_times = []
        for alg in self.algorithms:
            start_time =  dt.datetime.now()
            labels = alg.fit_predict(embeddings)
            end_time = dt.datetime.now()
            df[alg.__class__.__name__] = labels
            time_diff = end_time - start_time
            training_times.append(time_diff.total_seconds())

        return TrainingResults(
            embeddings=NormalizedEmbeddings(
                source=data.source,
                embeddings=df,
            ),
            models=[
                TrainedModel(
                    model=alg,
                    model_name=alg.__class__.__name__,
                    training_time=train_time_alg
                ) 
                for train_time_alg, alg in zip(training_times,self.algorithms)]
        )

    def settings(self):
        return {
            "trained_algorithms" : [{algorithm.__class__.__name__ : algorithm.get_params()} for algorithm in self.algorithms],
            "reduced_to" : self.reduce_to if self.reduce_to is not None else "None"
        }




