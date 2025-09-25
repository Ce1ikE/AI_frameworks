import json
import csv
from pathlib import Path
from PIL.Image import Image
import logging
import enum
import cv2
import pprint
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .FaceDetector import FaceDetector
from .FaceEmbedder import FaceEmbedder
from .FaceClassifier import FaceClassifier
from skl2onnx import to_onnx

# the Reporter class handles saving results and optionally saving cropped face images
# it will start by making a directory for the whole pipeline output
# and then save results depending on the configuration
# it can save:
# - bounding boxes (JSON,text,CSV)
# - embeddings (JSON,text,CSV)
# - cropped face images (same as input format, e.g. PNG,JPG)

# standard use case:
# ------------------
# it will use the image filename as a base for naming all saved files and directories
# and use the name_model of the detector + image filename for the root of the result directory 
# e.g. for an input image "image1.jpg" and detector "ViolaJones" it might save:
# - "ViolaJones_image1/" (directory)
# - "image1_results.json" (file with bounding boxes and embeddings)
# - "image1_annotated.jpg" (original image with bounding boxes drawn)
# - "cropped/" (directory with cropped face images)
# - "cropped/image1_cropped_0.jpg", "cropped/image1_cropped_1.jpg", ...
# creating the following file structure:
# output_dir/ViolaJones_image1/
#               /cropped/
#                   /image1_cropped_0.jpg
#                   /image1_cropped_1.jpg
#               /image1_results.csv
#               /image1_annotated.jpg
#               /model_settings.json

# bulk use case:
# --------------
# it will create a directory for the whole pipeline output with the model names
# e.g. for detector "ViolaJones", embedder "ArcFace" and classifier "SVC"
# - "ViolaJones_ArcFace_SVC_bulk/" (directory)
#   - "image1/" (subdirectory for image1.jpg)
#       - "cropped/" (directory with cropped face images)
#           - "image1_cropped_0.jpg"
#           - "image1_cropped_1.jpg"
#       - "image1_results.csv"
#       - "image1_annotated.jpg"
#       - "model_settings.json"
#   - "image2/" (subdirectory for image2.jpg)
#       - "cropped/" (directory with cropped face images)
#           - "image2_cropped_0.jpg"
#       - "image2_results.csv"
#       - "image2_annotated.jpg"
#       - "model_settings.json"
#   - "compiled_results.csv" (all results compiled into one CSV file)

class OutputFormat(enum.Enum):
    JSON = "json"
    CSV = "csv"
    TXT = "txt"


# https://scikit-learn.org/stable/model_persistence.html#model-persistence
class ModelFormat(enum.Enum):
    # I prefer ONNX for interoperability
    ONNX = "onnx" 
    # both joblib and pickle are part of the Python ecosystem but requires the same environment asx the training environment
    # which could be a problem later on
    JOBLIB = "joblib" 
    PICKLE = "pickle" 


@dataclass
class ReporterConfig:
    """Configuration class for Reporter output settings."""
    output_dir: Path
    prefix_save_dir: str = ""

    save_annotated_image: bool = True
    save_cropped_faces: bool = True
    save_model: bool = True
    save_model_settings: bool = True
    save_image_results_to_file: bool = True
    save_compiled_results: bool = True
    
    save_model_settings_format: OutputFormat = OutputFormat.JSON
    save_model_format: ModelFormat = ModelFormat.ONNX
    save_image_results_to_file_format: OutputFormat = OutputFormat.CSV
    save_compiled_results_format: OutputFormat = OutputFormat.CSV

    @property
    def is_saving_enabled(self) -> bool:
        """Check if any saving operation is enabled."""
        return (self.save_annotated_image or self.save_cropped_faces or 
                self.save_model or self.save_model_settings or 
                self.save_image_results_to_file or self.save_compiled_results)


class Reporter:
    logger = logging.getLogger(__name__)

    def __init__(self, config: ReporterConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir_result = None
        self.bulk_mode = False

    def _create_model_suffix(self, detector: FaceDetector, embedder: FaceEmbedder, classifier: FaceClassifier) -> str:
        """Create a suffix string from model names."""
        parts = [
            detector.get_name(),
            embedder.get_name() if embedder else "NoEmbedder",
            classifier.get_name() if classifier else "NoClassifier"
        ]
        return "_".join(parts)

    def setup_bulk_mode(self, detector: FaceDetector, embedder: FaceEmbedder, classifier: FaceClassifier):
        """Initialize bulk processing mode with a shared output directory."""
        self.bulk_mode = True
        model_suffix = self._create_model_suffix(detector, embedder, classifier)
        self.output_dir = self.output_dir / f"{model_suffix}_bulk"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created bulk output directory: {self.output_dir}")

    def setup_output_directory(self, detector: FaceDetector, embedder: FaceEmbedder, 
                             classifier: FaceClassifier, image_path: Path):
        """Setup output directory for current processing context."""
        if self.bulk_mode:
            # For bulk mode, create subdirectory for each image
            self.output_dir_result = self.output_dir / image_path.stem
        else:
            # For single mode, create directory with model info and image name
            model_suffix = self._create_model_suffix(detector, embedder, classifier)
            prefix = f"{self.config.prefix_save_dir}_" if self.config.prefix_save_dir else ""
            self.output_dir_result = self.output_dir / f"{prefix}{model_suffix}_{image_path.stem}"
        
        self.output_dir_result.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created output directory: {self.output_dir_result}")

    def compile_all_results(self):
        """Compile all CSV results from subdirectories in output_dir into a single file (bulk mode only)."""
        if not self.bulk_mode:
            self.logger.info("Not in bulk mode. Skipping results compilation.")
            return
        if not self.config.save_compiled_results:
            self.logger.info("Compiling results is disabled in the configuration.")
            return

        csv_files = list(self.output_dir.rglob("*.csv"))
        if not csv_files:
            self.logger.warning("No CSV files found to compile.")
            return
            
        compiled_data = pd.concat([pd.read_csv(filepath) for filepath in csv_files], ignore_index=True)
        output_path = self.output_dir / "compiled_results.csv"
        compiled_data.to_csv(output_path, index=False)
        self.logger.info(f"Compiled {len(csv_files)} CSV files into {output_path} with {len(compiled_data)} total entries.")
                
    def save(
        self, 
        detector: FaceDetector, 
        embedder: FaceEmbedder, 
        classifier: FaceClassifier,
        image: Image, 
        image_path: Path, 
        results: list[dict], 
        cropped_faces: list[Image] = None
    ):
        """Save all enabled outputs for the current processing results."""
        if not self.config.is_saving_enabled:
            self.logger.info("Saving is disabled in the configuration. No results will be saved.")
            return

        # Setup output directory
        self.setup_output_directory(detector, embedder, classifier, image_path)

        saving_errors = []
        try:
            # Save results to file
            if self.config.save_image_results_to_file and results:
                filename = f"{image_path.stem}_results.{self.config.save_image_results_to_file_format.value}"
                self.save_to_file(filename, results)
        except Exception as e:
            self.logger.error(f"Failed to save results to file: {e}")
            saving_errors.append(e)

        try:
            # Save cropped face images
            if self.config.save_cropped_faces and cropped_faces:
                self.save_img(image_path, "cropped", cropped_faces)
        except Exception as e:
            self.logger.error(f"Failed to save cropped face images: {e}")
            saving_errors.append(e)

        try:
            # Save annotated original image
            if self.config.save_annotated_image:
                self.save_annotated_image(detector, embedder, classifier, image_path, image, results)
        except Exception as e:
            self.logger.error(f"Failed to save annotated original image: {e}")
            saving_errors.append(e)

        try:
            # Save model settings
            if self.config.save_model_settings:
                self.save_model_settings(detector, embedder, classifier)
        except Exception as e:
            self.logger.error(f"Failed to save model settings: {e}")
            saving_errors.append(e)

        if saving_errors:
            self.logger.error(f"Encountered {len(saving_errors)} errors during saving operations.")
            raise RuntimeError(f"Errors occurred during saving: {saving_errors}")

    def _write_json(self, file_path: Path, data: list[dict]):
        """Write data to JSON file."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _write_csv(self, file_path: Path, data: list[dict]):
        """Write data to CSV file."""
        with open(file_path, "w+", newline="") as f:
            writer = csv.DictWriter(f, data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    
    def _write_txt(self, file_path: Path, data: list[dict]):
        """Write data to text file."""
        with open(file_path, "w") as f:
            for index, entry in enumerate(data):
                f.write(f"Result {index}:\n")
                for key, value in entry.items():
                    f.write(f"  {key}: {value}\n")

    def save_to_file(self, filename: str, data: list[dict], format: OutputFormat = None):
        """Save data to file in the specified format."""
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")

        if not data:
            self.logger.warning(f"No data to save. Skipping saving to file {filename}.")
            return
        
        format = format or self.config.save_image_results_to_file_format
        output_file = self.output_dir_result / filename
        
        format_handlers = {
            OutputFormat.JSON: self._write_json,
            OutputFormat.CSV: self._write_csv, 
            OutputFormat.TXT: self._write_txt
        }
        
        handler = format_handlers.get(format)
        if handler:
            handler(output_file, data)
            self.logger.info(f"Saved results to {output_file}")
        else:
            raise ValueError(f"Unsupported output format: {format}")

    def save_img(self, image_path: Path, face_type: str, faces: list[Image]):
        """Save cropped face images to a subdirectory."""
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")
        
        images_dir = self.output_dir_result / face_type
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for i, face in enumerate(faces):
            face_image_path = images_dir / f"{image_path.stem}_{face_type}_{i}{image_path.suffix}"
            cv2.imwrite(str(face_image_path), face)
        
        self.logger.info(f"Saved {len(faces)} {face_type} face images to {images_dir}")

    def save_model_settings(self, detector: FaceDetector, embedder: FaceEmbedder, classifier: FaceClassifier):
        """Save model configuration settings."""
        settings_data = [
            {"detector": detector.settings()},
            {"embedder": embedder.settings() if embedder else "NoEmbedder"},
            {"classifier": classifier.settings() if classifier else "NoClassifier"},
        ]
        self.save_to_file("model_settings.json", settings_data, OutputFormat.JSON)

    def save_model(self, classifier: FaceClassifier, train_data: np.ndarray):
        """Save trained model to ONNX format."""
        if not self.config.save_model:
            return
        # training set, can be None, it is used to infered the input types (initial_types)
        onx = to_onnx(classifier.model, train_data[:1].astype(np.float32), target_opset=12)
        file_path = self.output_dir_result / f"{classifier.get_name()}.onnx"

        with open(file_path, "wb") as f:
            f.write(onx.SerializeToString())
        
        self.logger.info(f"Saved model to {file_path}")

    def save_annotated_image(
        self, 
        detector: FaceDetector, 
        embedder: FaceEmbedder, 
        classifier: FaceClassifier, 
        image_path: Path, 
        image: Image, 
        results: list[dict]
    ):
        """Save original image with face detection annotations."""
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")
        
        # draw bounding boxes and landmarks for each detected face
        for entry in results:
            x, y, width, height = entry["bbox"]
            
            # draw the bounding box at the detected location
            cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 6)
            # draw landmarks (if available)
            if (landmarks := entry.get("landmarks")) is not None:
                for face_landmark in landmarks:
                    for landmark in face_landmark:
                        cv2.circle(image, tuple(landmark[:2]), 2, (0, 255, 0), 3)
            # draw confidence score (if available)
            if (score := entry.get("score")) is not None:
                # convert to scalar float safely
                if isinstance(score, np.ndarray):
                    if score.size == 1:  # length-1 array
                        score = float(score.item())
                    else:  # multi-element array: pick first element
                        score = float(score.flatten()[0])
                else:
                    score = float(score)

                cv2.putText(image, f"{score:.2f}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        # add model information text
        model_info = [
            f"Detector: {detector.__class__.__name__}",
            f"Embedder: {embedder.__class__.__name__ if embedder else 'N/A'}",
            f"Classifier: {classifier.__class__.__name__ if classifier else 'N/A'}",
            f"Faces: {len(results)}"
        ]
        for i, info in enumerate(model_info):
            cv2.putText(image, info, (10, 30 + i * 50), cv2.FONT_HERSHEY_COMPLEX, 
                       2, (255, 255, 255), 5, cv2.LINE_AA)
        # save annotated image
        output_path = self.output_dir_result / f"{image_path.stem}_annotated{image_path.suffix}"
        cv2.imwrite(str(output_path), image)
        self.logger.info(f"Saved annotated image to {output_path}")
