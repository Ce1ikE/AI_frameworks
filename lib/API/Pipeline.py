from pathlib import Path
from .FaceDetector import FaceDetector
from .FaceEmbedder import FaceEmbedder
from .Preprocessor import Preprocessor
from .Reporter import Reporter

class Pipeline:
    def __init__(
        self, 
        reporter: Reporter,
        preprocessor: Preprocessor,
        detector: FaceDetector, 
        embedder: FaceEmbedder = None,
    ):
        self.detector = detector
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.reporter = reporter

        if self.preprocessor is None:
            raise ValueError("Preprocessor is required for the pipeline.")
        if self.detector is None:
            raise ValueError("FaceDetector is required for the pipeline.")
        if self.reporter is None:
            raise ValueError("Reporter is required for the pipeline.")
    
        self.do_embed = embedder is not None
    
    def process(self, image_path: Path):
        # pipeline steps:
        # 1. Load image without any preprocessing
        # 2. Detect faces and return bounding boxes
        # 3. For each detected face:
        #    3.1 Crop the face
        #    3.2 Preprocess the face (resize, normalize, etc.)
        #    3.3 Generate embedding for the face (if embedder is provided)
        #    3.4 Collect results (bounding box, embedding, etc.)
        # 4. Report/Save results (save to file, etc.) (!!! if enabled)

        # 1.
        image = self.preprocessor.load(image_path)
        # 2.
        bboxes = self.detector.detect_faces(image)

        results = []
        preprocessed_faces = []
        cropped_faces = []
        # 3.
        for bbox in bboxes:
            # 3.1
            face = self.preprocessor.crop(image, bbox)
            cropped_faces.append(face)
            # 3.2
            preprocessed_face = self.preprocessor.resize(face)
            preprocessed_faces.append(preprocessed_face)
            # 3.3
            embedding = None
            if self.embedder is not None:
                embedding = self.embedder.embed_face(preprocessed_face)

            results.append({"bbox": bbox, "embedding": embedding})

        self.reporter.save(
            image,
            image_path, 
            results, 
            preprocessed_faces, 
            cropped_faces
        )