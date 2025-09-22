from pathlib import Path
from .FaceDetector import FaceDetector
from .FaceEmbedder import FaceEmbedder
from .Preprocessor import Preprocessor
from .Reporter import Reporter
import logging

class Pipeline:
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        reporter: Reporter,
        detector: FaceDetector, 
        embedder: FaceEmbedder = None,
    ):
        self.detector = detector
        self.embedder = embedder
        self.reporter = reporter

        if self.detector is None:
            raise ValueError("FaceDetector is required for the pipeline.")
        if self.reporter is None:
            raise ValueError("Reporter is required for the pipeline.")
    
        self.do_embed = embedder is not None

    def run(self, image_path: Path):
        self.process(image_path)
    
    def process(self, image_path: Path):
        # pipeline steps:
        # 1. Load image without any preprocessing
        # 2. Detect faces and return bounding boxes
        # 3. For each detected face:
        #    3.1 Crop the face
        #    3.2 Generate embedding for the face (if embedder is provided)
        #    3.3 Collect results (bounding box, embedding, etc.)
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

        # 4.
        self.logger.debug(f"Reporting results for image: {image_path}")
        self.reporter.save(
            detector=self.detector,
            embedder=self.embedder if self.do_embed else None,
            image=image,
            image_path=image_path, 
            results=results, 
            cropped_faces=cropped_faces
        )