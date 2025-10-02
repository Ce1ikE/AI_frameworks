import json
import csv
from pathlib import Path
from PIL.Image import Image
import logging
import enum
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .FaceDetector import FaceDetector
from .FaceEmbedder import FaceEmbedder
from .FaceClassifier import FaceClassifier
from ..face_classification.KMeansClassifier import KMeansClassifier
from .Preprocessor import Preprocessor
from skl2onnx import to_onnx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import time

# the Reporter class handles saving results of a pipeline 
# it will start by making a directory for the whole pipeline output
# and then save results depending on the configuration
# it can save:
# - bounding boxes (JSON,text,CSV)
# - embeddings (JSON,text,CSV)
# - cropped face images (same as input format, e.g. PNG,JPG)
# - annotated original image (JPG,PNG)
# - model settings (JSON)
# - etc...

# standard use case:
# ------------------
# it will use the image filename as a output base for naming all saved files and directories
# e.g. for an input image "image1.jpg" it will create the following structure:
# /img_<timestamp> (directory)
#       /image1_results.json (file with bounding boxes and embeddings)
#       /image1_annotated.jpg (original image with bounding boxes drawn)
#       /model_settings.json (file with model settings)
#       /cosine_similarity_matrix_image1.csv (file with cosine similarity matrix)
#       /cosine_similarity_matrix_image1.html (HTML visualization of the cosine similarity matrix)
#       /cosine_similarity_matrix_image1.svg (SVG visualization of the cosine similarity matrix)
#       /embeddings/ (directory with .npy files for each embedding)
#           /embedding_image1_0.npy (embedding for face 0)
#           /embedding_image1_1.npy (embedding for face 1)
#           /...
#       /cropped/ (directory with cropped face images)
#           /cropped/image1_cropped_0.jpg (cropped face 0)
#           /cropped/image1_cropped_1.jpg (cropped face 1)
#           /...

# bulk use case:
# --------------
# once the bulk mode is enabled by calling setup_bulk_mode()
# it will create a shared directory for the whole pipeline output 
# and then create a subdirectory for each image with the image name
# e.g. for input images "image1.jpg" and "image2.jpg" it will create the following structure:
# /Bulk_<timestamp> (directory)
#       /img_image1 (directory)
#           /image1_results.json (file with bounding boxes and embeddings)
#           /image1_annotated.jpg (original image with bounding boxes drawn)
#           /model_settings.json (file with model settings)
#           /cosine_similarity_matrix_image1.csv (file with cosine similarity matrix)
#           /cosine_similarity_matrix_image1.html (HTML visualization of the cosine similarity matrix)
#           /cosine_similarity_matrix_image1.svg (SVG visualization of the cosine similarity matrix)
#           /embeddings/ (directory with .npy files for each embedding)
#               /embedding_image1_0.npy (embedding for face 0)
#               /embedding_image1_1.npy (embedding for face 1)
#               /...
#           /cropped/ (directory with cropped face images)
#               /cropped/image1_cropped_0.jpg (cropped face 0)
#               /cropped/image1_cropped_1.jpg (cropped face 1)
#               /...
#       /img_image2 (directory)
#           /image2_results.json (file with bounding boxes and embeddings)
#           /image2_annotated.jpg (original image with bounding boxes drawn)
#           /model_settings.json (file with model settings)
#           /cosine_similarity_matrix_image2.csv (file with cosine similarity matrix)
#           /cosine_similarity_matrix_image2.html (HTML visualization of the cosine similarity matrix)
#           /cosine_similarity_matrix_image2.svg (SVG visualization of the cosine similarity matrix)
#           /embeddings/ (directory with .npy files for each embedding)
#               /embedding_image2_0.npy (embedding for face 0)
#               /embedding_image2_1.npy (embedding for face 1)
#               /...
#           /cropped/ (directory with cropped face images)
#               /cropped/image2_cropped_0.jpg (cropped face 0)
#               /cropped/image2_cropped_1.jpg (cropped face 1)
#               /...
#       /compiled_results.parquet (file with all results compiled into a single file)
#       /batch_summary.json (file with summary of the batch processing)
#       /tsne_visualization_bulk.svg (t-SNE visualization of all embeddings in the batch)
#       /tsne_visualization_bulk.html (HTML t-SNE visualization of all embeddings in the batch)
#       /...

# training use case:
# ------------------
# once the training mode is enabled by calling setup_training_output_directory()
# it will create a shared directory for the whole training output
# e.g. for training a classifier it will create the following structure:
# /Training_<timestamp> (directory)
#       /test_summary.json (file with summary of the test processing)
#       /KMeansClassifier.onnx (trained model in ONNX format)
#       /model_settings.json (file with model settings)
#       /cluster_centers.parquet (file with cluster centers in parquet format)
#       /cluster_centers.npy (file with cluster centers in numpy format)
#       /inertia.svg (plot of inertia over number of clusters)
#       /silhouette_scores.svg (plot of silhouette scores over number of clusters)
#       /...



class OutputFormat(enum.Enum):
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    BIN = "bin"
    PARQUET = "parquet"


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

    save_annotated_image: bool = True
    save_cropped_faces: bool = True
    save_model: bool = True
    save_model_settings: bool = True
    save_image_results_to_file: bool = True
    save_compiled_results: bool = True
    save_cosine_similarity_matrix: bool = True
    save_cosine_similarity_matrix_visualization: bool = True
    save_tsne_visualization: bool = True
    
    save_model_settings_format: OutputFormat = OutputFormat.JSON
    save_model_format: ModelFormat = ModelFormat.ONNX
    save_image_results_to_file_format: OutputFormat = OutputFormat.CSV
    save_compiled_results_format: OutputFormat = OutputFormat.PARQUET

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
        self.bulk_mode = False

    # ------------------ Directory and File Management ------------------ #

    def setup_output_directory(
        self, 
        image_path: Path
    ):
        """Setup output directory for current processing result (single image)."""
        if self.bulk_mode:
            # for bulk processing, create subdirectory for each image with the image name
            self.output_dir_result = self.output_dir / image_path.stem
        else:
            # for single processing, create directory 
            img_dir = f"img_{image_path.stem}"
            self.output_dir_result = self.output_dir / img_dir

        self.output_dir_result.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created output directory: {self.output_dir_result}")

    def setup_bulk_output_directory(self):
        """Initialize bulk processing mode with a shared output directory. (shared for all images)"""
        self.bulk_mode = True
        bulk_dir = f"Bulk_{time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = self.output_dir / bulk_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created bulk output directory: {self.output_dir}")

    def setup_training_output_directory(self):
        """Setup output directory for training a classifier."""
        train_dir = f"Training_{time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = self.output_dir / train_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created training output directory: {self.output_dir}")

    # ------------------ Saving Methods ------------------ #

    def compile_all_results(self, output_dir: Path = None):
        """Compile all embedding results from subdirectories in output_dir into a single file (bulk mode only)."""
        if not self.bulk_mode:
            self.logger.info("Not in bulk mode. Skipping results compilation.")
            return
        
        if output_dir is not None:
            self.output_dir = output_dir

        json_files = list(self.output_dir.rglob("*results.json"))
        if not json_files:
            self.logger.warning("No JSON result files found.")
            return
        try:
            # we load all JSON files and combine them into a single list of entries
            # in each JSON file, there is a entry which points to the .npy file with the embedding
            # if this entry is present, we load the .npy file and add the embedding to the entry
            # after that we looked at all JSON files we combine them into a single parquet file
            all_entries = []
            for f in json_files:
                with open(f, "r") as fp:
                    # 1 JSON file contains a list of entries
                    # each entry is a dict with keys: bounding_box, score, embedding, label, etc.
                    # for a single face detected in an image
                    entries = json.load(fp)  
                    for entry in entries:
                        # if present load embeddings from .npy
                        if "embedding_file" in entry:
                            emb_path = self.output_dir / entry["embedding_file"]
                            if emb_path.exists():
                                try:
                                    entry["embedding"] = np.load(emb_path).flatten()
                                except Exception as e:
                                    self.logger.error(f"Failed to load embedding from {emb_path}: {e}")        
                        all_entries.append(entry)

            compiled_data_filename = self.output_dir / "compiled_results.parquet"
            # the reason we save to parquet is that it supports complex data types like lists and numpy arrays
            # and is more efficient than CSV or JSON for large datasets with many embeddings
            # pandas also supports reading and writing parquet files natively so it is easy to work with
            self._write_parquet(compiled_data_filename, all_entries)
            self.logger.info(f"Compiled {len(all_entries)} results into {compiled_data_filename}")
        except Exception as e:
            self.logger.error(f"Failed to compile results: {e}")
            return
        
            # save t-SNE visualization for compiled results
        try:
            if self.config.save_tsne_visualization:
                self.save_tsne_visualization(compiled_data_filename, all_entries)
        except Exception as e:
            self.logger.error(f"Failed to save additional compiled results visualizations: {e}")
            return
        
    def save_batch_summary(self, batch_result: dict):
        """Save a summary of the batch processing results."""
        if not self.output_dir:
            raise RuntimeError("Output directory not set. Call setup_bulk_mode() first.")
        
        if not batch_result:
            self.logger.warning("No batch results to save.")
            return
        
        for paths in ["succeeded_paths", "failed_paths"]:
            if paths in batch_result:
                batch_result[paths] = [Path(p).relative_to(self.output_dir.parent.parent).as_posix() for p in batch_result[paths]]

        summary_file = self.output_dir / "batch_summary.json"
        self._write_json(summary_file, [batch_result])
    
        if self.config.save_compiled_results:
            self.compile_all_results()

    def save_test_summary(
        self, 
        test_result: dict,
        classifier: KMeansClassifier, 
        train_data: np.ndarray,
        silhouette_scores: list[float] = None,
        inertias: list[float] = None,
    ):
        if not self.output_dir:
            raise RuntimeError("Output directory not set. Call setup_bulk_mode() first.")
        if not test_result:
            self.logger.warning("No test results to save.")
            return
        
        self.setup_training_output_directory()

        summary_file = self.output_dir / "test_summary.json"
        self._write_json(summary_file,[test_result])

        if self.config.save_model and classifier:
            self.save_model(self.output_dir, classifier, train_data)

            # compile all cluster centers into a single .parquet file and a single .npy file
            if classifier.cluster_centers is not None:
                np.save(self.output_dir / "cluster_centers.npy", classifier.cluster_centers)
                data = []
                for center in classifier.cluster_centers:
                    data.append({"cluster_center": center})
                path_parquet = self.output_dir / "cluster_centers.parquet"
                self._write_parquet(path_parquet, data)

            # graph inertia and silhouette scores if available
            if inertias is not None and len(inertias) > 0:
                path_inertia = self.output_dir / "inertia.svg"
                self.save_model_inertia(inertias)
                self.logger.info(f"Saved inertia to {path_inertia}")

            if silhouette_scores is not None and len(silhouette_scores) > 0:
                path_silhouette = self.output_dir / "silhouette_scores.svg"
                self.save_model_silhouette(silhouette_scores)
                self.logger.info(f"Saved silhouette scores to {path_silhouette}")

    def save_processed_image_results(
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

        # setup output directory
        # depending on if we are in bulk mode or not
        # a subdirectory for each image is created in bulk mode
        self.setup_output_directory(image_path)

        saving_errors = []
        try:
            # Save cosine similarity matrix
            if self.config.save_cosine_similarity_matrix and results:
                self.save_cosine_similarity_matrix(image_path, results)
        except Exception as e:
            self.logger.error(f"Failed to save cosine similarity matrix: {e}")
            saving_errors.append(e)


        try:
            # Save cropped face images
            if self.config.save_cropped_faces and cropped_faces:
                self.save_face_images(image_path, "cropped", cropped_faces, results)
        except Exception as e:
            self.logger.error(f"Failed to save cropped face images: {e}")
            saving_errors.append(e)

        try:
            # Save results to file
            if self.config.save_image_results_to_file and results:
                self.save_results_to_file(image_path, results)
        except Exception as e:
            self.logger.error(f"Failed to save results to file: {e}")
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
    
    def save_to_file(self, filepath: Path, data: list[dict], format: OutputFormat = None):
        """Save data to file in the specified format."""
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")

        if not data:
            self.logger.warning(f"No data to save. Skipping saving to file {filepath}.")
            return
        
        format = format or self.config.save_image_results_to_file_format
        output_file = filepath if filepath.is_absolute() else self.output_dir / filepath
        
        format_handlers = {
            OutputFormat.JSON: self._write_json,
            OutputFormat.CSV: self._write_csv, 
            OutputFormat.TXT: self._write_txt,
            OutputFormat.BIN: self._write_bin,
        }
        
        handler = format_handlers.get(format)
        if handler is None:
            raise ValueError(f"Unsupported output format: {format}")
        handler(output_file, data)
        self.logger.info(f"Saved results to {output_file}")

    def save_results_to_file(self, image_path: Path, results: list[dict]):
        # results from a image are saved in a JSON file 
        # the embeddings are saved in a separate .npy file which is located in a subdirectory
        # we add a reference to the .npy file in the JSON file

        # each entry in results is a dict 
        # of each detected face in the image
        for entry in results:
            if "embedding" in entry and entry["embedding"] is not None and len(entry["embedding"]) > 0:
                # create subdirectory for embeddings
                embeddings_dir = self.output_dir_result / "embeddings"
                embeddings_dir.mkdir(parents=True, exist_ok=True)
                # save embedding to .npy file
                embedding_filename = f"embedding_{image_path.stem}_{results.index(entry)}.npy"
                embedding_path = embeddings_dir / embedding_filename
                np.save(embedding_path, entry["embedding"])
                # replace embedding in results with reference to file
                entry["embedding_file"] = embedding_path.relative_to(self.output_dir_result.parent).as_posix()
                del entry["embedding"]

            # save metadata separately
            meta_file_path = self.output_dir_result / f"{image_path.stem}_results.{self.config.save_image_results_to_file_format.value}"
        self.save_to_file(meta_file_path, results, format=self.config.save_image_results_to_file_format)

    def save_face_images(self, image_path: Path, face_type: str, faces: list[Image], results: list[dict]):
        """Save cropped face images to a subdirectory."""
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")
        
        images_dir = self.output_dir_result / face_type
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for i, face in enumerate(faces):
            face_image_path = images_dir / f"{image_path.stem}_{face_type}_{i}{image_path.suffix}"
            cv2.imwrite(str(face_image_path), face)
            # add reference to cropped face image in results
            results[i]["face_image"] = face_image_path.relative_to(self.output_dir_result.parent).as_posix()

        self.logger.info(f"Saved {len(faces)} {face_type} face images to {images_dir}")

    def save_model_settings(self, detector: FaceDetector, embedder: FaceEmbedder, classifier: FaceClassifier):
        """Save model configuration settings.(single image)"""
        settings_data = [
            {"detector": detector.settings()},
            {"embedder": embedder.settings() if embedder else "NoEmbedder"},
            {"classifier": classifier.settings() if classifier else "NoClassifier"},
        ]
        model_path = self.output_dir_result / "model_settings.json"
        self.save_to_file(model_path, settings_data, OutputFormat.JSON)

    def save_model(self, path: Path, classifier: FaceClassifier, train_data: np.ndarray):
        """Save trained model to ONNX format."""
        # https://onnx.ai/sklearn-onnx/index.html
        if not self.config.save_model:
            return
        # training set, can be None, it is used to infered the input types (initial_types)
        onx = to_onnx(classifier.model, train_data[:1].astype(np.float32), target_opset=12)
        file_path = path / f"{classifier.get_name()}.onnx"

        with open(file_path, "wb") as f:
            f.write(onx.SerializeToString())
        
        self.logger.info(f"Saved model to {file_path}")

    def save_cosine_similarity_matrix(self, image_path: Path, results: list[dict]):
        """Compute and save cosine similarity matrix of embeddings. of the detected faces."""
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")
        
        df = pd.DataFrame(results)
        if "embedding" in df.columns:
            # Extract embedding columns
            embeddings = df["embedding"].to_numpy()
            if len(embeddings) < 2:
                self.logger.warning("Not enough embeddings to compute cosine similarity matrix.")
                return
        else:
            self.logger.warning(f"Invalid results format. Expected embedding column. got {df.columns}")
            return
        
        # creates a 2D numpy array from the list of embeddings
        # [n_samples, n_features(512 for ArcFace)]
        embeddings_matrix = np.vstack(embeddings)
        # we compute the cosine similarity of each embedding against each other by 
        # using the formula: cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
        # a = emb
        # b = emb_T (transposed)
        cosine_similarity_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)
        # normalize the cosine similarity matrix to be in the range [0, 1]
        norms = np.linalg.norm(embeddings_matrix, axis=1)
        cosine_similarity_matrix /= norms[:, np.newaxis]
        cosine_similarity_matrix /= norms[np.newaxis, :]
        # clip values to be in the range [0, 1]
        cosine_similarity_matrix = np.clip(cosine_similarity_matrix, min=0.0, max=1.0)
        # Save the cosine similarity matrix as a CSV file
        similarity_file = self.output_dir_result / f"cosine_similarity_matrix_{image_path.stem}.csv"
        df = pd.DataFrame(cosine_similarity_matrix)
        df.to_csv(similarity_file, index=False)
        
        if self.config.save_cosine_similarity_matrix_visualization and cosine_similarity_matrix.shape[0] >= 2:
            self.save_cosine_similarity_matrix_visualization(similarity_file, df)
        self.logger.info(f"Saved cosine similarity matrix to {similarity_file}")

    def save_tsne_visualization(self, path: Path, data: list[dict]):
        """Compute and save t-SNE visualization of embeddings."""
        df = pd.DataFrame(data)
        if "embedding" in df.columns:
            # Extract embedding columns
            embeddings = df["embedding"].to_numpy()
            if len(embeddings) < 2:
                self.logger.warning("Not enough embedding columns to compute t-SNE visualization.")
                return
        else:
            self.logger.warning(f"Invalid results format. Expected embedding column. got {df.columns}")
            return
        
        embeddings_matrix = np.vstack(embeddings)
        perplexity  = min(30, (embeddings_matrix.shape[0] - 1) // 3)
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity, 
            random_state=42
        )
        tsne_results = tsne.fit_transform(embeddings_matrix)
        # If face_images are provided, we'll use them as markers in custom plot
        # https://learnopencv.com/t-sne-for-feature-visualization/
        plot_size = 1080
        face_size = 50
        tsne_plot = 255 * np.ones((plot_size, plot_size, 3), dtype=np.uint8)
        # normalize t-SNE coordinates to fit within the plot area
        x_min, x_max = tsne_results[:, 0].min(), tsne_results[:, 0].max()
        y_min, y_max = tsne_results[:, 1].min(), tsne_results[:, 1].max()
        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        for image_path, x, y in zip(df.get("face_image", []), tsne_results[:, 0], tsne_results[:, 1]):
            full_image_path = self.output_dir / image_path if image_path else None
            if full_image_path and full_image_path.exists():
                face_image = cv2.imread(str(full_image_path))
                face_image = cv2.resize(face_image, (face_size, face_size))
                # Normalize and scale coordinates
                x_norm = int((x - x_min) / x_range * (plot_size - face_size))
                y_norm = int((y - y_min) / y_range * (plot_size - face_size))
                # Ensure indices are within bounds
                if 0 <= x_norm <= plot_size - face_size and 0 <= y_norm <= plot_size - face_size:
                    tsne_plot[y_norm:y_norm+face_size, x_norm:x_norm+face_size] = face_image
        tsne_file = self.output_dir / f"tsne_visualization_{path.stem}_with_faces.png"
        cv2.imwrite(tsne_file.as_posix(), tsne_plot)
        self.logger.info(f"Saved t-SNE visualization with face images to {tsne_file}")

        # -------------- plotting 2D -------------- #
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1])
        plt.title(f"t-SNE Visualization of Embeddings for {path.stem}")
        tsne_file = self.output_dir / f"tsne_visualization_{path.stem}.svg"
        plt.savefig(tsne_file, format="svg")
        plt.close()
        # -------------- plotting 2D -------------- #
        
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity, 
            random_state=42
        )
        projections = tsne.fit_transform(embeddings_matrix)
        # -------------- plotting HTML 3D -------------- #
        fig = px.scatter_3d(
            projections,
            x=0, 
            y=1, 
            z=2,
            title=f"t-SNE 3D Visualization of Embeddings for {path.stem}"
        )
        fig.update_traces(
            marker={
                "size": 5
            }
        )
        tsne_file = self.output_dir / f"tsne_visualization_{path.stem}.html"
        fig.write_html(tsne_file)

        self.logger.info(f"Saved t-SNE visualization to {tsne_file}")

    def save_cosine_similarity_matrix_visualization(self, path: Path, df: pd.DataFrame):
        """Compute and save cosine similarity matrix visualization of embeddings."""
        if not self.output_dir_result:
            raise RuntimeError("Output directory structure not created. Call create_output_dir_structure() first.")
        if df.shape[0] < 2:
            self.logger.warning("Not enough embeddings to compute cosine similarity matrix visualization.")
            return
        
        # -------------- plotting HTML -------------- #
        fig = px.imshow(df.values,
            labels=dict(x="Samples", y="Samples", color="Cosine similarity"),
            x=df.index, y=df.index,
            color_continuous_scale="RdBu_r",
            aspect="auto")
        fig.update_layout(title=f"Cosine Similarity Matrix for {path.stem}")
        similarity_file = self.output_dir_result / f"cosine_similarity_matrix_{path.stem}.html"
        fig.write_html(similarity_file)
        # -------------- plotting HTML -------------- #
        
        # -------------- plotting SVG -------------- #
        annotations = np.where(
            np.abs(df) > 0.5,
            df.round(2).astype(dtype=str),
            ""
        )
        plt.figure(figsize=(10, 8))
        plt.title(f"Cosine Similarity Matrix for {path.stem}")
        sns.heatmap(
            data=df, 
            cmap="coolwarm", 
            square=True, 
            annot=annotations,
            annot_kws={"fontsize":8},
            linewidths=0.5, 
            fmt="s"
        )
        similarity_file = self.output_dir_result / f"cosine_similarity_matrix_{path.stem}.svg"
        plt.savefig(similarity_file, format="svg")
        plt.close()
        # -------------- plotting SVG -------------- #
        self.logger.info(f"Saved cosine similarity matrix visualization to {similarity_file}")

    def save_model_inertia(self, inertia: list[float]):
        """Save inertia plot for KMeans clustering."""
        if not self.output_dir:
            raise RuntimeError("Output directory not set. Call setup_training_output_directory() first.")
        if len(inertia) < 2:
            self.logger.warning("Not enough inertia values to plot.")
            return
        
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(inertia) + 1), inertia, marker='o')
        plt.title("KMeans Inertia")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.xticks(range(1, len(inertia) + 1))
        plt.grid()
        inertia_file = self.output_dir / "inertia.svg"
        plt.savefig(inertia_file, format="svg")
        plt.close()
        self.logger.info(f"Saved inertia plot to {inertia_file}")

    def save_model_silhouette(self, silhouette_scores: list[float]):
        """Save silhouette scores plot for KMeans clustering."""
        if not self.output_dir:
            raise RuntimeError("Output directory not set. Call setup_training_output_directory() first.")
        if len(silhouette_scores) < 2:
            self.logger.warning("Not enough silhouette scores to plot.")
            return
        
        plt.figure(figsize=(10, 8))
        plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
        plt.title("KMeans Silhouette Scores")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.xticks(range(2, len(silhouette_scores) + 2))
        plt.grid()
        silhouette_file = self.output_dir / "silhouette_scores.svg"
        plt.savefig(silhouette_file, format="svg")
        plt.close()
        self.logger.info(f"Saved silhouette scores plot to {silhouette_file}")

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
        
        labels = []
        # draw bounding boxes and landmarks for each detected face
        for entry in results:
            x, y, width, height = entry["bbox"]
            
            # draw the bounding box at the detected location
            cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 6)
            # draw landmarks (if available)
            if (landmarks := entry.get("landmarks")) is not None:
                for face_landmark in landmarks:
                    cv2.circle(image, tuple(face_landmark[:2]), 10, (0, 255, 0), 4)
            # draw confidence score (if available)
            if (score := entry.get("score")) is not None:
                # convert to scalar float safely
                if isinstance(score, np.ndarray):
                    if score.size == 1:
                        score = float(score.item())
                    else:
                        score = float(score.flatten()[0])
                else:
                    score = float(score)

                cv2.putText(
                    image, f"{score:.2f}", (x + 10, y - 10), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), 
                    thickness=2, lineType=cv2.LINE_AA
                )
            if (identity := entry.get("label")) is not None:
                cv2.putText(
                    image, str(identity), (x + width - 10, y), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), 
                    thickness=2, lineType=cv2.LINE_AA
                )
                labels.append(str(identity))
        # add model information text
        model_info = [
            f"Detector: {detector.__class__.__name__}",
            f"Embedder: {embedder.__class__.__name__ if embedder else 'N/A'}",
            f"Classifier: {classifier.__class__.__name__ if classifier else 'N/A'}",
            f"Faces: {len(results)}",
            f"Identities: {', '.join(set(labels)) if labels else 'N/A'}"
        ]
        offset = 65
        init_y = 100
        init_x = 20
        for i, info in enumerate(model_info):
            cv2.putText(
                image, info, 
                org=(init_x, init_y + i * offset), 
                fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                fontScale=2, color=(255, 255, 255), 
                thickness=5, lineType=cv2.LINE_AA
            )
        # save annotated image
        output_path = self.output_dir_result / f"{image_path.stem}_annotated{image_path.suffix}"
        self._write_jpg(output_path, image)
        self.logger.info(f"Saved annotated image to {output_path}")

    # ------------------ Internal Helper Methods ------------------ #

    def _serialize_value(self, value):
        """Ensure numpy arrays/lists are JSON-compatible."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (np.floating,np.float32,np.float64)):
            return float(value)
        if isinstance(value, (np.integer,np.int32,np.int64)):
            return int(value)
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return value
    
    def _write_json(self, file_path: Path, data: list[dict]):
        """Write data to JSON file."""
        serializable_data = [
            {k: self._serialize_value(v) for k, v in entry.items()}
            for entry in data
        ]
        with open(file_path, "w") as f:
            json.dump(serializable_data, f, indent=2)
    
    def _write_csv(self, file_path: Path, data: list[dict]):
        """Write data to CSV file."""
        self.logger.warning("Writing to CSV will be deprecated in future versions. Consider using Parquet or JSON format instead.")
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            for entry in data:
                row = {k: json.dumps(self._serialize_value(v)) if isinstance(v, (list, np.ndarray)) else v
                       for k, v in entry.items()}
                writer.writerow(row)
    
    def _write_txt(self, file_path: Path, data: list[dict]):
        """Write data to text file."""
        with open(file_path, "w") as f:
            for index, entry in enumerate(data):
                f.write(f"Result {index}:\n")
                for key, value in entry.items():
                    f.write(f"  {key}: {self._serialize_value(value)}\n")

    def _write_bin(self, file_path: Path, data: list[dict]):
        """Write data to binary file using numpy's savez."""
        keys = data[0].keys()
        dict_of_lists = {key: [] for key in keys}
        for entry in data:
            for key in keys:
                dict_of_lists[key].append(self._serialize_value(entry.get(key)))
        np.savez(file_path, **dict_of_lists)

    def _write_parquet(self, file_path: Path, data: list[dict]):
        """Write data to Parquet file using pandas."""
        df = pd.DataFrame(data)
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html
        df.to_parquet(file_path, index=False)

    def _write_jpg(self, file_path: Path, image: Image):
        """Write image to JPG file."""
        cv2.imwrite(str(file_path), image)