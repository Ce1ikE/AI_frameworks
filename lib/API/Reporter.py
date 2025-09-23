import json
import csv
from pathlib import Path
from PIL.Image import Image
import logging
import enum
import cv2
import pprint
from .FaceDetector import FaceDetector
from .FaceEmbedder import FaceEmbedder
from .FaceClassifier import FaceClassifier

# the Reporter class handles saving results and optionally saving cropped and preprocessed face images
# it will start by making a directory for the whole pipeline output
# and then save results depending on the configuration
# it can save:
# - bounding boxes (JSON,text,CSV)
# - embeddings (JSON,text,CSV)
# - cropped face images (same as input format, e.g. PNG,JPG)
# - preprocessed face images (same as input format, e.g. PNG,JPG)

# it will use the image filename as a base for naming all saved files and directories
# and use the name_model of the detector + image filename for the root of the result directory 
# e.g. for an input image "image1.jpg" and detector "ViolaJones" it might save:
# - "ViolaJones_image1/" (directory)
# - "image1_results.json" (file with bounding boxes and embeddings)
# - "image1_annotated.jpg" (original image with bounding boxes drawn)
# - "cropped/" (directory with cropped face images)
# - "cropped/image1_cropped_0.jpg", "cropped/image1_cropped_1.jpg", ...
# - "preprocessed/" (directory with preprocessed face images)
# - "preprocessed/image1_preprocessed_0.jpg", "preprocessed/image1_preprocessed_1.jpg", ...

class OutputFormat(enum.Enum):
    JSON = "json"
    CSV = "csv"
    TXT = "txt"


class Reporter:
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        output_dir: Path, 
        prefix_save_dir: str = None,
        save_original_image: bool = False,
        save_to_file: bool = True,
        save_format: OutputFormat = OutputFormat.CSV,
        save_cropped_faces: bool = False, 
        save_model_settings: bool = True,
    ):
        self.output_dir = output_dir
        self.output_dir_result = None
        self.prefix_save_dir = prefix_save_dir if prefix_save_dir else ""

        self.save_original_image = save_original_image
        self.save_to_file = save_to_file
        self.save_format = save_format
        self.save_cropped_faces = save_cropped_faces
        self.save_model_settings = save_model_settings
        self.enable_saving = False

        if self.save_cropped_faces or self.save_to_file or self.save_original_image or self.save_model_settings:
            self.enable_saving = True

    def create_output_bulk_dir_structure(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        classifier: FaceClassifier,
    ):
        dir_suffix = detector.get_name()
        dir_suffix += "_" + (embedder.get_name() if embedder is not None else "NoEmbedder")
        dir_suffix += "_" + (classifier.get_name() if classifier is not None else "NoClassifier")
        self.output_dir = self.output_dir / f"{dir_suffix}_bulk"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created bulk output directory: {self.output_dir}")

    def create_output_dir_structure(self, dir_suffix: str, image_path: Path):
        self.output_dir_result = Path(self.output_dir / f"{self.prefix_save_dir}_{dir_suffix}_{image_path.stem}")
        self.output_dir_result.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created output directory: {self.output_dir_result}")

    def save(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        classifier: FaceClassifier,
        image: Image,
        image_path: Path, 
        results: list[dict],
        cropped_faces: list[Image],
    ):
        if not self.enable_saving:
            self.logger.info("Saving is disabled in the configuration. No results will be saved.")
            return

        dir_suffix = detector.get_name()
        dir_suffix += "_" + (embedder.get_name() if embedder is not None else "NoEmbedder")
        dir_suffix += "_" + (classifier.get_name() if classifier is not None else "NoClassifier")
        self.create_output_dir_structure(dir_suffix, image_path)

        if self.save_to_file:
            self.save__to_file(
                self.output_dir_result,
                f"{image_path.stem}_results.{self.save_format.value}", 
                image_path,
                results
            )

        if self.save_cropped_faces:
            self.save__faces(
                image_path, 
                "cropped", 
                cropped_faces
            )

        if self.save_original_image:
            self.save__original_image(
                detector, 
                embedder, 
                classifier, 
                image_path, 
                image, 
                results
            )

        if self.save_model_settings:
            self.save__model_settings(
                detector,
                embedder,
                classifier
            )

    def save__to_file(
        self,
        dir_output: Path,
        filename: str,
        image_path: Path,
        results: list[dict],
        save_format: OutputFormat = None 
    ):
        if not dir_output:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")

        if not results or len(results) == 0:
            self.logger.warning(f"No results to save for image {image_path}. Skipping saving to file.")
            return
        
        if save_format is None:
            save_format = self.save_format

        output_file = dir_output / filename 
        if save_format == OutputFormat.JSON:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
        elif save_format == OutputFormat.CSV:
            with open(output_file, "w+", newline="") as f:
                writer = csv.DictWriter(f, results[0].keys())
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
        elif save_format == OutputFormat.TXT:
            with open(output_file, "w") as f:
                for index, entry in enumerate(results):
                    f.write(f"Result {index}:\n")
                    for key, value in entry.items():
                        f.write(f"  {key}: {value}\n")

        self.logger.info(f"Saved results to {output_file}")

    def save__faces(
        self, 
        image_path: Path, 
        face_type: str,
        faces: list[Image], 
    ):
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")
        # create face type directory
        face_images_dir = Path(self.output_dir_result / face_type)
        face_images_dir.mkdir(parents=True, exist_ok=True)
        for i, face in enumerate(faces):
            face_image_path = face_images_dir / f"{image_path.stem}_{face_type}_{i}.{image_path.suffix}"
            cv2.imwrite(face_image_path, face)
        self.logger.info(f"Saved {len(faces)} {face_type} face images to {face_images_dir}")    

    def save__report(self):
        pass

    def save__model_settings(
        self, 
        detector: FaceDetector, 
        embedder: FaceEmbedder, 
        classifier: FaceClassifier
    ):
        self.save__to_file(
            dir_output=self.output_dir_result,
            filename="model_settings",
            image_path=None,
            results=[
                {"detector": detector.settings()},
                {"embedder": embedder.settings() if embedder else "NoEmbedder"},
                {"classifier": classifier.settings() if classifier else "NoClassifier"},
            ],
            save_format=OutputFormat.JSON
        )

    def save__original_image(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        classifier: FaceClassifier,
        image_path: Path,
        image: Image,
        results
    ):
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")
        # save original image with annotations: 
        # - bounding boxes 
        # - detector 
        # - embedder (if available)
        # - classifier (if available)
        # - number of faces detected
        # - not names yet

        for entry in results:
            (x, y, width, height), _ = entry["bbox"], entry.get("embedding", None)
            cv2.rectangle(
                image, 
                (x, y),
                (x + width, y + height),
                (255, 0, 0),
                6
            )

        info_text = f"Detector: {detector.__class__.__name__}\n"
        info_text += f"Embedder: {embedder.__class__.__name__ if embedder else 'N/A'}\n"
        info_text += f"Classifier: {classifier.__class__.__name__ if classifier else 'N/A'}\n"
        info_text += f"Faces: {len(results)}\n"
        cv2.putText(
            img=image,
            text=info_text,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        cv2.imwrite(self.output_dir_result / f"{image_path.stem}_annotated.{image_path.suffix}", image)
        self.logger.info(f"Saved original image with annotations to {self.output_dir_result / f'{image_path.stem}_annotated.{image_path.suffix}'}")
