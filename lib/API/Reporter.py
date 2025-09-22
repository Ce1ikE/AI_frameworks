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

# the Reporter class handles saving results and optionally saving cropped and preprocessed face images
# it will start by making a directory for the whole pipeline output
# and then save results depending on the configuration
# it can save:
# - bounding boxes (JSON,text,CSV)
# - embeddings (JSON,text,CSV)
# - cropped face images (same as input format, e.g. PNG,JPG)
# - preprocessed face images (same as input format, e.g. PNG,JPG)
# - TODO : make a summary report (text,HTML,PDF)

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
        save_original_image: bool = False,
        save_to_file: bool = True,
        save_format: OutputFormat = OutputFormat.CSV,
        save_cropped_faces: bool = False, 
    ):
        self.output_dir = output_dir
        self.output_dir_result = None

        self.save_original_image = save_original_image
        self.save_to_file = save_to_file
        self.save_format = save_format
        self.save_cropped_faces = save_cropped_faces
        self.enable_saving = False

        if self.save_cropped_faces or self.save_to_file or self.save_original_image:
            self.enable_saving = True

    def create_output_dir_structure(self, dir_suffix: str, image_path: Path):
        self.output_dir_result = Path(self.output_dir / f"{dir_suffix}_{image_path.stem}")
        self.output_dir_result.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created output directory: {self.output_dir_result}")

    def save(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        image: Image,
        image_path: Path, 
        results: list[dict],
        cropped_faces: list[Image],
    ):
        if not self.enable_saving:
            self.logger.info("Saving is disabled in the configuration. No results will be saved.")
            return

        dir_suffix = detector.get_name() + "_" + (embedder.get_name() if embedder is not None else "NoEmbedder")
        self.create_output_dir_structure(dir_suffix, image_path)

        if self.save_to_file:
            self.save__to_file(image_path, results)
        if self.save_cropped_faces:
            self.save__faces(image_path, "cropped", cropped_faces)
        if self.save_original_image:
            self.save__original_image(detector,embedder,image_path, image, results)

    def save__to_file(
        self,
        image_path: Path,
        results: list[dict],
    ):
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")

        if not results or len(results) == 0:
            self.logger.warning(f"No results to save for image {image_path}. Skipping saving to file.")
            return

        output_file = self.output_dir_result / f"{image_path.stem}_results.{self.save_format.value}"
        if self.save_format == OutputFormat.JSON:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
        elif self.save_format == OutputFormat.CSV:
            with open(output_file, "w+", newline="") as f:
                writer = csv.DictWriter(f, results[0].keys())
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
        elif self.save_format == OutputFormat.TXT:
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

    def save__original_image(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
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

        info_text = f"""
            Detector: {detector.__class__.__name__}
            Embedder: {embedder.__class__.__name__ if embedder else 'N/A'}
            Faces: {len(results)}
        """
        cv2.putText(
            image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.imwrite(self.output_dir_result / f"{image_path.stem}_annotated.{image_path.suffix}", image)
        self.logger.info(f"Saved original image with annotations to {self.output_dir_result / f'{image_path.stem}_annotated.{image_path.suffix}'}")
