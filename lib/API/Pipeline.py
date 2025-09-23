from pathlib import Path
from .FaceDetector import FaceDetector
from .FaceEmbedder import FaceEmbedder
from .FaceClassifier import FaceClassifier
from .Preprocessor import Preprocessor
from .Reporter import Reporter
import logging
# we're trying to imitate the sklearn pipeline design here
# by having a unified interface for the pipeline
# that takes in a detector, embedder, classifier and reporter
# and runs the pipeline of course this pipeline is very basic
# but the idea is that we can perform both inference and training
# and adapt the pipeline to different use cases (only detection, detection + embedding, detection + embedding + classification)

# https://github.com/serengil/retinaface
# "A modern face recognition pipeline consists of 4 common stages: 
# detect, align, normalize, represent and verify. 
# Experiments show that alignment increases the face recognition accuracy almost 1%. 
# Here, retinaface can find the facial landmarks including eye coordinates. 
# In this way, it can apply alignment to detected faces with its extracting faces function.
# Notice that face recognition module of insightface project is ArcFace, and face detection module is RetinaFace. 
# ArcFace and RetinaFace pair is wrapped in deepface library for Python. 
# Consider to use deepface if you need an end-to-end face recognition pipeline."

# while i won't use deepface directly,
# i will use the same idea of having a pipeline with these stages
# where i have a detector (RetinaFace,Viola Jones, YuNet), an embedder/represent (ArcFace) and optionally a classifier (e.g. SVM, KNN, etc.)
# the only thing that i add is the automatic reporting/saving of results which i believe is very useful for practical applications
class Pipeline:
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        reporter: Reporter,
        detector: FaceDetector, 
        embedder: FaceEmbedder = None,
        classifier: FaceClassifier = None,
        bulk_mode: bool = False
    ):
        self.detector = detector
        self.embedder = embedder
        self.classifier = classifier
        self.reporter = reporter

        if bulk_mode and self.reporter is not None:
            self.reporter.create_output_bulk_dir_structure(
                self.detector, 
                self.embedder,
                self.classifier
            )

    def run(self, image_path: Path):
        self.process(image_path)

    def train(self, image_paths: list[Path], labels: list[str]):
        if self.classifier is None:
            raise ValueError("FaceClassifier is required for training pipeline.")

    def process(self, image_path: Path):
        if self.detector is None:
            raise ValueError("FaceDetector is required for the processing pipeline.")
        if self.reporter is None:
            raise ValueError("Reporter is required for the processing pipeline.")
        # pipeline steps:
        # 1. Load image without any preprocessing
        # 2. Detect faces and return bounding boxes
        # 3. For each detected face:
        #    3.1 Crop the face
        #    3.2 Generate embedding for the face (if embedder is provided)
        #    3.3 Collect results (bounding box, embedding, etc.)
        #    3.4 Classify the face (if classifier is provided)
        # 4. Report/Save results (save to file, etc.) (!!! if enabled)

        # 1.
        image = Preprocessor.load(image_path)
        self.logger.info(f"Processing image: {image_path}")
        # 2.
        bboxes = self.detector.detect_faces(image)
        self.logger.info(f"Detected bboxes: {len(bboxes)} in image: {image_path}")

        results = []
        cropped_faces = []
        # 3.
        for bbox in bboxes:
            # 3.1
            face = Preprocessor.crop(image, bbox)
            cropped_faces.append(face)
            
            embedding = None
            if self.embedder is not None:
                # 3.2
                embedding = self.embedder.embed_face(face)
            # 3.3
            results.append({"bbox": bbox, "embedding": embedding})
            if self.classifier is not None and embedding is not None:
                # 3.4
                label = self.classifier.classify_face(embedding)
                results[-1]["label"] = label

        # 4.
        self.logger.debug(f"Reporting results for image: {image_path}")
        self.reporter.save(
            detector=self.detector,
            embedder=self.embedder,
            classifier=self.classifier,
            image=image,
            image_path=image_path, 
            results=results, 
            cropped_faces=cropped_faces
        )