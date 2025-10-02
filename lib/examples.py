from .Core import Core
from .API.Pipeline import Pipeline
from .API.Preprocessor import Preprocessor
# detectors (img -> bboxes + optionally landmarks + scores)
from .face_detection.ViolaJones import ViolaJonesDetector , CascadeType
from uniface.constants import RetinaFaceWeights
from .face_detection.RetinaFace import RetinaFaceDetector
from .face_detection.YuNet import YuNetDetector
from .face_detection.SCRFD import SCRFDDetector
from .face_detection.HoG import HoGDetector
# embedders (face img -> embedding vector)
from .face_representation.ArcFace import ArcFaceEmbedder , ArcFaceWeights
# classifiers (embedding vector -> label)
from .face_classification.KMeansClassifier import KMeansClassifier
from .face_classification.LoadedClassifier import LoadedClassifier
from .face_classification.MetricClassifier import MetricClassifier , Metric

from pathlib import Path
import logging
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class Example(Enum):
    CONVERT = 1
    DETECT = 2
    DETECT_EMBED = 3
    DETECT_EMBED_CLASSIFY = 4
    COMPILE = 5
    TRAIN = 6
    INFERENCE = 7

    __all__ = [CONVERT, DETECT_EMBED, COMPILE, TRAIN, INFERENCE]

def convert_heic_to_jpg__pipeline(input_files: list[Path], delete_heic_files: bool = False):    
    for input_file in input_files:
        jpg_path = input_file.parent / (input_file.stem + ".jpg")
        # we don't want any duplicate jpg files lying around
        if jpg_path.is_file():
            jpg_path.unlink()

        Preprocessor.convert_heic_to_jpg(
            source=input_file,
            dest=jpg_path
        )

        if delete_heic_files:
            input_file.unlink()
        logger.info(f"Converted {input_file} to {jpg_path}")

def detect_faces__pipeline(core: Core):
    """
        A pipeline that only detects faces in images 
        and saves the results.
    """
    reporter = Pipeline.create_reporter(output_dir=core.paths.output)
    
    return Pipeline(
        reporter=reporter,
        detector=RetinaFaceDetector(
            model_name=RetinaFaceWeights.MNET_025,
            confidence_threshold=0.6,
            dynamic_size=True,
        ),
        embedder=None,
        classifier=None,
    )

def detect_embed_faces__pipeline(core: Core):
    """
        A pipeline that detects faces in images, 
        creates embeddings for those faces, 
        and saves the results.
    """
    reporter = Pipeline.create_reporter(output_dir=core.paths.output)
    
    return Pipeline(
        reporter=reporter,
        detector=RetinaFaceDetector(
            model_name=RetinaFaceWeights.MNET_025,
            confidence_threshold=0.6,
            dynamic_size=True,
        ),
        embedder=ArcFaceEmbedder(
            model_name=ArcFaceWeights.W600K_MBF
        ),
    )

def detect_embed_classify_faces__pipeline(core: Core, cluster_centers: np.ndarray):
    """
        A pipeline that detects faces in images, 
        creates embeddings for those faces, 
        classifies those embeddings,
        and saves the results.
    """
    reporter = Pipeline.create_reporter(output_dir=core.paths.output)
    
    return Pipeline(
        reporter=reporter,
        detector=RetinaFaceDetector(
            model_name=RetinaFaceWeights.MNET_025,
            confidence_threshold=0.6,
            dynamic_size=True,
        ),
        embedder=ArcFaceEmbedder(
            model_name=ArcFaceWeights.W600K_MBF
        ),
        classifier=MetricClassifier(
            cluster_centers=cluster_centers,
            metric=Metric.COSINE,
        ),
    )



def train_classifier__pipeline(core: Core):
    reporter = Pipeline.create_reporter(output_dir=core.paths.output)
    
    return Pipeline(
        reporter=reporter,
        detector=None,
        embedder=None,
        classifier=None,
    )



def compile_all_results__pipeline(core: Core,path: Path = None):
    reporter = Pipeline.create_reporter(
        output_dir=core.paths.output,
        save_cropped_faces=True,
        save_model_settings=True,
    )
    reporter.bulk_mode = True
    reporter.compile_all_results(path)

def inference__pipeline(core: Core):
    pass