import json
import csv
from pathlib import Path
from PIL import Image
import logging
import enum

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
# e.g. for an input image "image1.jpg", it might save:
# - "image1_results.json"

class OutputFormat(enum.Enum):
    JSON = "json"
    CSV = "csv"
    TXT = "txt"


class Reporter:
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        output_dir: Path, 
        save_to_file: bool = True,
        save_format: OutputFormat = OutputFormat.CSV,
        save_cropped_faces: bool = False, 
        save_preprocessed_faces: bool = False,
    ):
        self.output_dir = output_dir
        self.output_dir_result = None

        self.save_to_file = save_to_file
        self.save_format = save_format
        self.save_cropped_faces = save_cropped_faces
        self.save_preprocessed_faces = save_preprocessed_faces
        

    def create_output_dir_structure(
        self,
        image_path: Path
    ):
        self.output_dir_result = Path(self.output_dir / image_path.stem)
        self.output_dir_result.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created output directory: {self.output_dir_result}")

    def save(
        self, 
        image_path: Path, 
        results,
        format: OutputFormat = OutputFormat.JSON
    ):
        if self.save_to_file:
            self.save__to_file(image_path, results, format)
        if self.save_cropped_faces:
            self.save__cropped_face(image_path, results)
        if self.save_preprocessed_faces:
            self.save__preprocessed_face(image_path, results)

    def save__to_file(
        self,
        image_path: Path,
        results,
        format: OutputFormat = OutputFormat.JSON
    ):
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")

        output_file = self.output_dir_result / f"{image_path.stem}_results.{format.value}"
        if format == OutputFormat.JSON:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
        elif format == OutputFormat.CSV:
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "value"])
                for key, value in results.items():
                    writer.writerow([key, value])
        elif format == OutputFormat.TXT:
            with open(output_file, "w") as f:
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")
        self.logger.info(f"Saved results to {output_file}")

    def save__cropped_face(
        self, 
        image_path: Path, 
        face: Image.Image, 
        bbox
    ):
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")
        cropped_face_path = self.output_dir / f"{image_path.stem}_cropped_{bbox}.png"
        face.save(cropped_face_path)
        print(f"Saved cropped face to {cropped_face_path}")

    def save__preprocessed_face(
        self, 
        image_path: Path, 
        face: Image.Image, 
        bbox
    ):
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")
        preprocessed_face_path = self.output_dir / f"{image_path.stem}_preprocessed_{bbox}.png"
        face.save(preprocessed_face_path)
        print(f"Saved preprocessed face to {preprocessed_face_path}")
