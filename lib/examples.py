from .Core import Core
from .API.PathManager import PathManager
from .API.Pipeline import Pipeline
from .API.Reporter import Reporter, ReporterConfig, OutputFormat , ModelFormat
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

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def advanced_example_with_direct_config__pipeline(core: Core):
    """Example showing direct ReporterConfig usage for advanced users."""
    config = ReporterConfig(
        output_dir=Path(core.paths.output),
        prefix_save_dir="advanced_DE_faces_",
        save_annotated_image=True,
        save_cropped_faces=True,
        save_model=True,    
        save_model_settings=True,
        save_image_results_to_file=True,
        save_compiled_results=True,
        save_model_settings_format=OutputFormat.JSON,
        save_model_format=ModelFormat.ONNX,
        save_image_results_to_file_format=OutputFormat.CSV,
        save_compiled_results_format=OutputFormat.CSV
    )
    
    reporter = Reporter(config)
    
    return Pipeline(
        reporter=reporter,
        detector=RetinaFaceDetector(
            model_name=RetinaFaceWeights.MNET_025,
            confidence_threshold=0.6
        ),
        embedder=ArcFaceEmbedder(
            model_name=ArcFaceWeights.W600K_MBF
        ),
        classifier=None,
        bulk_mode=True,
    )

def detect_and_embed_faces__pipeline(core: Core):
    reporter = Pipeline.create_reporter(
        output_dir=core.paths.output,
        prefix_save_dir="DE_faces_",
        save_cropped_faces=True,
        save_model_settings=True,
    )
    
    return Pipeline(
        reporter=reporter,
        detector=RetinaFaceDetector(
            model_name=RetinaFaceWeights.MNET_025,
            confidence_threshold=0.6
        ),
        embedder=ArcFaceEmbedder(
            model_name=ArcFaceWeights.W600K_MBF
        ),
        classifier=None,
        bulk_mode=True,
    )

def detect_embed_and_classify_faces__pipeline(core: Core, classifier_path: Path):
    reporter = Pipeline.create_reporter(
        output_dir=core.paths.output,
        prefix_save_dir="DEC_faces_",
        save_cropped_faces=True,
        save_model_settings=True,
    )
    
    return Pipeline(
        reporter=reporter,
        detector=RetinaFaceDetector(
            model_name=RetinaFaceWeights.MNET_025,
            confidence_threshold=0.6
        ),
        embedder=ArcFaceEmbedder(
            model_name=ArcFaceWeights.W600K_MBF
        ),
        classifier=LoadedClassifier(
            model_path=classifier_path
        ),
        bulk_mode=True,
    )

def detect_faces__pipeline(core: Core):
    reporter = Pipeline.create_reporter(
        output_dir=core.paths.output,
        prefix_save_dir="D_faces_",
        save_cropped_faces=True,
        save_model_settings=True,
    )
    
    return Pipeline(
        reporter=reporter,
        detector=RetinaFaceDetector(
            model_name=RetinaFaceWeights.MNET_025,
            confidence_threshold=0.6
        ),
        embedder=None,
        classifier=None,
        bulk_mode=True,
    )

def train_classifier__pipeline(core: Core):
    reporter = Pipeline.create_reporter(
        output_dir=core.paths.output,
        prefix_save_dir="D_faces_",
        save_cropped_faces=True,
        save_model_settings=True,
    )
    
    return Pipeline(
        reporter=reporter,
        detector=None,
        embedder=None,
        classifier=KMeansClassifier(),
        bulk_mode=False,
    )

def convert_heic_to_jpg__pipeline(core: Core, input_files: list[Path], delete_heic_files: bool = False):    
    for input_file in input_files:
        jpg_path = core.paths.input / (input_file.stem + ".jpg")
        # we don't want any duplicate jpg files lying around
        if jpg_path.is_file():
            jpg_path.unlink()

        Preprocessor.convert_heic_to_jpg(heic_path=input_file,jpg_path=jpg_path)
        if delete_heic_files:
            input_file.unlink()
        core.logger.info(f"Converted {input_file} to {jpg_path}")