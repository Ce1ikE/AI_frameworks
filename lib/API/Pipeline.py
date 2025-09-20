from pathlib import Path
from .FaceDetector import FaceDetector
from .FaceEmbedder import FaceEmbedder
from .Preprocessor import Preprocessor
from .Reporter import Reporter

class Pipeline:
    def __init__(
        self, 
        detector: FaceDetector, 
        embedder: FaceEmbedder,
        preprocessor: Preprocessor,
        reporter: Reporter,
    ):
        self.detector = detector
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.reporter = reporter

    def process(self, image_path: Path):
        self.reporter.create_output_dir_structure(image_path)

        # pipeline steps:
        # 1. Load image without any preprocessing
        # 2. Detect faces and return bounding boxes
        # 3. For each detected face:
        #    3.1 Crop the face
        #       3.1.1 and optionally save the cropped face image
        #    3.2 Preprocess the face (resize, normalize, etc.)
        #       3.2.1 and optionally save the preprocessed face image
        #    3.3 Generate embedding for the face
        #    3.4 Collect results (bounding box, embedding, etc.)
        # 4. Report results (save to file, etc.)

        # 1.
        image = self.preprocessor.load(image_path)
        # 2.
        bboxes = self.detector.detect_faces(image)

        # 3.
        results = []
        for bbox in bboxes:
            # 3.1
            face = self.preprocessor.crop(image, bbox)
            # 3.2
            face = self.preprocessor.resize(face)
            # 3.3
            embedding = self.embedder.embed_face(face)
            results.append({"bbox": bbox, "embedding": embedding})
        
        self.reporter.save(image_path, results)
        return results