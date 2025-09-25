from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import numpy as np

from lib.face_classification.LoadedClassifier import LoadedClassifier
from .FaceDetector import FaceDetector
from .FaceEmbedder import FaceEmbedder
from .FaceClassifier import FaceClassifier
from .Preprocessor import Preprocessor
from .Reporter import ModelFormat, Reporter, ReporterConfig , OutputFormat
import logging
import time
from enum import Enum
import pandas as pd
from pandas import DataFrame
# https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/
from sklearn.cluster import KMeans
# https://www.geeksforgeeks.org/machine-learning/what-is-silhouette-score/
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


class AbortPipelineException(Exception):
    def __init__(self, message, errors):            
        super().__init__(message)
        self.errors = errors

class PipelineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ABORTED = "aborted"

class PipelineTask(Enum):
    DETECT = "detect"
    EMBED = "embed"
    CLASSIFY = "classify"
    TRAIN = "train"


# we're trying to imitate the sklearn pipeline design here
# by having a unified interface for the pipeline
# that takes in a detector, embedder, classifier and reporter
# and runs the pipeline of course this pipeline is very basic compared to what sklearn offers
# but the idea is that we can centralize both inference , training and reporting
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
# and where larger datasets are used, the reporting becomes essential

# the pipeline should be able to handle single images as well as batches of images
# therefore Pipeline offers a run() method for batch processing and a process() method for single image processing
# and a train() method for unsupervised training using clustering on face embeddings
# the pipeline should also be able to handle errors gracefully and continue processing other images 
# which means that FaceDetector, FaceEmbedder and FaceClassifier are not responsible for error handling !!

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
        """Initialize Pipeline with components.
        
        Args:
            reporter: Reporter instance (use Pipeline.create_reporter() for easy setup)
            detector: Face detection model
            embedder: Face embedding model (optional)
            classifier: Face classification model (optional)
            bulk_mode: Whether to process multiple images in bulk mode
        """
        self.detector = detector
        self.embedder = embedder
        self.classifier = classifier
        self.reporter = reporter
        self.status = PipelineStatus.IDLE
        self.task = None

        if bulk_mode and self.reporter is not None:
            self.reporter.setup_bulk_mode(
                self.detector, 
                self.embedder,
                self.classifier
            )

    # -------------------------------------- Static methods for easy setup (less boilerplate) -------------------------------------- #

    @classmethod
    def create_reporter(
        cls,
        output_dir: Path,
        prefix_save_dir: str = "",
        save_cropped_faces: bool = True,
        save_model_settings: bool = True,
        save_model: bool = True,
    ) -> Reporter:
        """Create a Reporter with simplified configuration.
        
        Args:
            output_dir: Base directory for saving outputs
            prefix_save_dir: Prefix for output directory names
            save_cropped_faces: Save cropped face images
            save_model_settings: Save model configuration
            save_model: Save trained models
            
        Returns:
            Configured Reporter instance
        """
        
        config = ReporterConfig(
            output_dir=output_dir,
            prefix_save_dir=prefix_save_dir,
            save_annotated_image=True,
            save_cropped_faces=save_cropped_faces,
            save_model=save_model,
            save_model_settings=save_model_settings,
            save_image_results_to_file=True,
            save_compiled_results=True,
            save_model_settings_format=OutputFormat.JSON,
            save_model_format=ModelFormat.ONNX,
            save_image_results_to_file_format=OutputFormat.CSV,
            save_compiled_results_format=OutputFormat.CSV
        )
        
        return Reporter(config)

    # -------------------------------------- Pipeline public methods -------------------------------------- #

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline configuration."""
        return {
            "detector": {
                "class": self.detector.__class__.__name__ if self.detector else None,
                "available": self.detector is not None,
                "settings": self.detector.settings() if self.detector else {}
            },
            "embedder": {
                "class": self.embedder.__class__.__name__ if self.embedder else None,
                "available": self.embedder is not None,
                "settings": self.embedder.settings() if self.embedder else {}
            },
            "classifier": {
                "class": self.classifier.__class__.__name__ if self.classifier else None,
                "available": self.classifier is not None,
                "settings": self.classifier.settings() if self.classifier else {}
            },
            "reporter": {
                "class": self.reporter.__class__.__name__ if self.reporter else None,
                "available": self.reporter is not None,
                "bulk_mode": getattr(self.reporter, 'bulk_mode', False) if self.reporter else False
            },
            "capabilities": {
                "can_detect": self.detector is not None,
                "can_embed": self.embedder is not None,
                "can_classify": self.classifier is not None and self.embedder is not None,
                "can_train": all([self.detector, self.embedder, self.classifier]) and not isinstance(self.classifier, LoadedClassifier),
                "can_save": self.reporter is not None
            }
        }

    def validate_pipeline_for_task(self, task: str) -> bool:
        """
        Validate that the pipeline has the required components for a specific task.
        
        Args:
            task: Task type ('detect', 'embed', 'classify', 'train')
            
        Returns:
            bool: True if pipeline can perform the task
        """
        info = self.get_pipeline_info()
        
        task_requirements = {
            PipelineTask.DETECT: info['capabilities']['can_detect'],
            PipelineTask.EMBED: info['capabilities']['can_embed'],
            PipelineTask.CLASSIFY: info['capabilities']['can_classify'],
            PipelineTask.TRAIN: info['capabilities']['can_train']
        }
        
        return task_requirements.get(task, False)

    # -------------------------------------- Core pipeline processing methods -------------------------------------- #
    def inference(self, image: Any, metric: str = "cosine", cluster_centers: List[np.ndarray] = None) -> dict:
        """
        Run inference on a single image through the pipeline.
        this pipeline does not handle loading or validating the image

        in order to classify the embeddings, an metric must be provided \n
        possible metrics are: \n
            "euclidean" \n https://www.geeksforgeeks.org/dsa/pairs-with-same-manhattan-and-euclidean-distance/ \n
            "cosine" \n https://www.geeksforgeeks.org/dbms/cosine-similarity/ \n
            "manhattan" \n https://www.geeksforgeeks.org/dsa/pairs-with-same-manhattan-and-euclidean-distance/ \n

        Args:
            image: The input image to process.

        Returns:
            dict: Inference results containing detected faces, embeddings, and classifications.
        """
        self._validate_components_for_inference()

        # Run detection
        bboxes, landmarks, scores = self.detector.detect_faces(image)

        face_results = []
        # Run embeddings
        for bbox in bboxes:
            embedding = self.embedder.embed_face(image, bbox) if self.embedder else None

            face_result = {
                "bbox": bbox,
                "landmarks": landmarks,
                "score": scores,
                "embedding": embedding,
                "label": None
            }

            # Classify face if classifier and a embedding are available
            if embedding is not None and self.classifier is not None:
                try:
                    label = self.classifier.predict(embedding)
                    face_result["label"] = label
                except Exception as e:
                    self.logger.warning(f"Failed to classify face: {e}")
            # Classify using custom cluster centers and metric if no classifier is available
            elif metric and cluster_centers is not None and embedding is not None:
                try:
                    distances = []
                    for center in cluster_centers:
                        if metric == "euclidean":
                            dist = np.linalg.norm(embedding - center)
                        elif metric == "cosine":
                            dist = 1 - np.dot(embedding, center) / (np.linalg.norm(embedding) * np.linalg.norm(center))
                        elif metric == "manhattan":
                            dist = np.sum(np.abs(embedding - center))
                        distances.append(dist)
                    face_result["label"] = np.argmin(distances)
                except Exception as e:
                    self.logger.warning(f"Failed to classify face: {e}")
            else:
                self.logger.debug("No classifier or custom cluster centers provided; skipping classification.")
                face_result["label"] = None
            face_results.append(face_result)
        return face_results

    def process(self, image_path: Path) -> dict:
        """
        Process a single image through the face detection pipeline.
        
        Args:
            image_path: Path to the image file to process
            
        Returns:
            dict: Processing results containing faces found, errors, and metadata
            
        Raises:
            ValueError: If required components are missing or image is invalid
            FileNotFoundError: If image file doesn't exist
        """
        # Pipeline steps:
        # ---------------
        # 1. Load and validate image
        # 2. Detect faces
        # 3. Process each detected face
        #     a. Crop face
        #     b. Embed face (if embedder available)
        #     c. Classify face (if classifier available)
        # 4. Save results
        # Validate components and inputs
        self._validate_components_for_processing()
        
        processing_result = {
            "image_path": str(image_path),
            "faces_detected": 0,
            "faces_processed": 0,
            "errors": [],
            "success": False
        }
        
        try:
            self.logger.info(f"Processing image: {image_path}")
            
            # 1) load and validate image
            image = self._load_and_validate_image(image_path)
            
            # 2) detect faces
            bboxes, landmarks, scores = self.detector.detect_faces(image)
            processing_result["faces_detected"] = len(bboxes)
            
            if len(bboxes) == 0:
                self.logger.info(f"No faces detected in image: {image_path}")
                processing_result["success"] = True
                return processing_result
            
            self.logger.info(f"Detected {len(bboxes)} faces in image: {image_path}")
            
            # 3) process each detected face
            results = []
            cropped_faces = []
            
            for i, bbox in enumerate(bboxes):
                # a) crop, embed, classify and collect results
                face, face_result = self._process_single_face(image, bbox, landmarks, scores, i)
                
                if face is not None and face_result is not None:
                    cropped_faces.append(face)
                    results.append(face_result)
                    processing_result["faces_processed"] += 1
                else:
                    processing_result["errors"].append(f"Failed to process face {i}")

            # 4) save results if any faces were successfully processed
            if results:
                self.logger.debug(f"Saving results for {len(results)} faces from image: {image_path}")
                self.reporter.save(
                    detector=self.detector,
                    embedder=self.embedder,
                    classifier=self.classifier,
                    image=image,
                    image_path=image_path,
                    results=results,
                    cropped_faces=cropped_faces,
                )
            
            processing_result["success"] = True
            self.logger.info(f"Successfully processed image: {image_path} "
                           f"({processing_result['faces_processed']}/{processing_result['faces_detected']} faces)")
            
        except Exception as e:
            error_msg = f"Pipeline error processing {image_path}: {str(e)}"
            self.logger.error(error_msg)
            processing_result["errors"].append(error_msg)
            processing_result["success"] = False
            
        return processing_result

    def run(
        self, 
        image_paths: list[Path], 
        continue_on_error: bool = True, 
        progress_callback=None,
    ) -> dict:
        """
        Process multiple images through the pipeline.

        Args:
            image_paths: List of image file paths to process
            continue_on_error: If True, continue processing other images when one fails
            progress_callback: Optional callback function called with (current, total, current_path)

            
        Returns:
            dict: Batch processing results with summary statistics
        """
        # Pipeline steps:
        # ---------------
        #     1. Validate input image paths
        #     2. Initialize batch results
        #     3. For each image
        #         3.1 Call progress callback if provided
        #         3.2 Process single image
        #             3.2.1 Load and validate image
        #             3.2.2 Detect faces
        #             3.2.3 Process each detected face
        #                 3.2.3.1 Crop face
        #                 3.2.3.2 Embed face (if embedder available)
        #                 3.2.3.3 Classify face (if classifier available)
        #             3.2.4 Save results
        #         3.3 Aggregate results
        #     4. Compile results if in bulk mode
        #     5. Log summary

        # 1) Validate
        if not image_paths:
            raise ValueError("image_paths cannot be empty")
        
        self.logger.info(f"Starting batch processing of {len(image_paths)} images")

        # 2) Initialize batch results
        batch_result = {
            "total_images": len(image_paths),
            "processed_images": 0,
            "failed_images": 0,
            "total_faces_detected": 0,
            "total_faces_processed": 0,
            "errors": [],
            "failed_paths": [],
            "success": True
        }

        # 3) Process each image
        for i, image_path in enumerate(image_paths):
            # 4) Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(i + 1, len(image_paths), image_path)
                except Exception as e:
                    self.logger.warning(f"Progress callback failed: {e}")
            
            try:
                # 5) Process single image
                result = self.process(image_path)

                # 6) Aggregate results
                if result["success"]:
                    batch_result["processed_images"] += 1
                    batch_result["total_faces_detected"] += result["faces_detected"]
                    batch_result["total_faces_processed"] += result["faces_processed"]
                else:
                    batch_result["failed_images"] += 1
                    batch_result["failed_paths"].append(str(image_path))
                    batch_result["errors"].extend(result["errors"])
                    
                    if not continue_on_error:
                        batch_result["success"] = False
                        self.logger.error(f"Stopping batch processing due to error in {image_path}")
                        break
                        
            except Exception as e:
                error_msg = f"Unexpected error processing {image_path}: {str(e)}"
                self.logger.error(error_msg)
                batch_result["failed_images"] += 1
                batch_result["failed_paths"].append(str(image_path))
                batch_result["errors"].append(error_msg)
                
                if not continue_on_error:
                    batch_result["success"] = False
                    break

        # 4) Compile results if in bulk mode
        if self.reporter.bulk_mode:
            try:
                self.reporter.compile_all_results()
            except Exception as e:
                self.logger.warning(f"Failed to compile batch results: {e}")

        # 5) Log summary
        self.logger.info(f"Batch processing completed: "
                        f"{batch_result['processed_images']}/{batch_result['total_images']} images successful, "
                        f"{batch_result['total_faces_processed']} faces processed")
        
        if batch_result["failed_images"] > 0:
            self.logger.warning(f"{batch_result['failed_images']} images failed to process")
        
        return batch_result

    def train(
        self, 
        X: DataFrame, 
        n_clusters: Optional[int] = None, 
        max_clusters: int = 20,
        split_ratio: float = 0.8,
        random_state: int = 42
    ) -> dict:
        """
        Train the classifier using unsupervised clustering on face embeddings.
        Face embeddings come in as DataFrame compiled from the process() or run() method.
        Only 1 classifier is supported for now, which must be a clustering algorithm
            - KMeans \n https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        number of clusters can be specified or estimated automatically using the elbow method and silhouette analysis \n
        https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

        Args:
            X: DataFrame with 'embedding' column containing face embeddings
            n_clusters: Number of clusters to create. If None, will attempt to estimate
            max_clusters: Maximum number of clusters to consider when estimating optimal clusters
            split_ratio: Ratio to split data into training and evaluation sets
            random_state: Random seed for reproducible results
            
        Returns:
            dict: Training results including cluster info, metrics, and embeddings
            
        Raises:
            ValueError: If required components are missing or no embeddings extracted
        """
        # pipeline steps:
        # ---------------
        # 1. Validate components and inputs
        # 2. Determine optimal number of clusters if not provided
        # 3. Train classifier with optimal clusters
        # 4. Evaluate clustering quality
        # 5. Save training results and model

        # 1) Validate components and inputs
        self._validate_components_for_training()
        self._validate_train_data(X)
        embeddings_array = np.vstack(X["embedding"].to_numpy())

        train_embeddings_X, test_embeddings_X = train_test_split(embeddings_array, test_size=1-split_ratio, random_state=random_state)

        self.logger.info(f"Starting unsupervised training pipeline with {len(train_embeddings_X)} training samples and {len(test_embeddings_X)} evaluation samples")

        # initialize training results
        training_result = {
            "clustering_method": self.classifier.__class__.__name__.lower(),
            "n_clusters_requested": n_clusters,
            "n_clusters_found": None,
            "processed_embeddings": 0,
            "total_embeddings": len(train_embeddings_X) + len(test_embeddings_X),
            "train_embeddings": len(train_embeddings_X),
            "test_embeddings": len(test_embeddings_X),
            "silhouette_score": None,
            "inertia": None,
            # cluster centers are required for classification during inference
            "cluster_centers": None,
            "cluster_labels": None,
            "errors": [],
            "success": False,
        }
        
        try:
            # 2) Determine optimal number of clusters if not provided
            if n_clusters is None:
                n_clusters , inertias = self._estimate_optimal_clusters(train_embeddings_X, max_clusters=min(max_clusters, len(train_embeddings_X)//2))
                self.logger.info(f"Estimated optimal number of clusters: {n_clusters}")
                training_result["inertia"] = inertias
            training_result["n_clusters_found"] = n_clusters
            
            # 3) Train classifier with n clusters
            self.logger.info(f"Training classifier with {n_clusters} clusters...")
            self.classifier.train(train_embeddings_X, n_clusters=n_clusters, random_state=random_state)

            # 4) Evaluate clustering quality
            results_evaluation = self.classifier.evaluate(test_embeddings_X)
            X["cluster_centers"] = results_evaluation.get("cluster_centers", None)
            X["cluster_labels"] = results_evaluation.get("cluster_labels", None)
            X["silhouette_score"] = results_evaluation.get("silhouette_score", None)

            # 5) Save training results and model
            if self.reporter.config.save_model and self.reporter.config.is_saving_enabled:
                try:
                    self.reporter.save_model(self.classifier, test_embeddings_X)
                except Exception as e:
                    self.logger.warning(f"Failed to save model: {e}")
            
            training_result["processed_embeddings"] = len(train_embeddings_X) + len(test_embeddings_X)
            training_result["success"] = True
            self.logger.info(f"Unsupervised training completed successfully!")
            
        except Exception as e:
            error_msg = f"Unsupervised training pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            training_result["errors"].append(error_msg)
            training_result["success"] = False

        finally:
            self.reporter.save_to_file("training_summary.json", training_result, OutputFormat.JSON)

        return training_result

    # -------------------------------------- Helper methods for internal processing -------------------------------------- # 

    # -------------------------------- Clustering helpers -------------------------------------- #
    def _estimate_optimal_clusters(self, embeddings: pd.Series, max_clusters: int = 20) -> tuple[int, list[float]]:
        """Estimate optimal number of clusters using elbow method and silhouette analysis."""        
        if len(embeddings) < 2:
            return 1
        if len(embeddings) < max_clusters:
            max_clusters = len(embeddings) - 1
            
        # we try different cluster numbers and evaluate the silhouette score
        silhouette_scores = []
        inertias = []
        K_range = range(2, min(max_clusters + 1, len(embeddings)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            silhouette_scores.append(silhouette_score(embeddings, cluster_labels,metric="euclidean"))
            inertias.append(kmeans.inertia_)
        
        # find elbow point by looking for max silhouette score (closest to 1 is better)
        if silhouette_scores:
            optimal_k = K_range[np.argmax(silhouette_scores)]
            return (optimal_k , inertias)
        
        # fallback
        return (max(2, len(embeddings) // 10) , inertias)

    # -------------------------------- Pipeline processing helpers -------------------------------------- #
    def _validate_components_for_inference(self):
        """Validate that required (detector , embedder , classifier) components are available for inference."""
        if self.detector is None:
            raise ValueError("FaceDetector is required for the inference pipeline")
        if self.embedder is None:
            raise ValueError("FaceEmbedder is required for the inference pipeline")

    def _validate_components_for_processing(self):
        """Validate that required (embedder and classifier are optional) components are available for processing."""
        if self.detector is None:
            raise ValueError("FaceDetector is required for the processing pipeline")
        if self.reporter is None:
            raise ValueError("Reporter is required for the processing pipeline")

    def _validate_components_for_training(self):
        """Validate that required components are available for training."""
        if self.classifier is None:
            raise ValueError("FaceClassifier is required for the training pipeline")
        if self.reporter is None:
            raise ValueError("Reporter is required for the training pipeline")

    def _load_and_validate_image(self, image_path: Path):
        """Load image and validate it exists and is readable."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Preprocessor.load(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            return image
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")

    def _validate_train_data(self, X_train: DataFrame):
        """Validate training data DataFrame. The Pipeline should NOT be responsible for data cleaning."""
        if X_train is None or X_train.empty:
            raise ValueError("Training data X cannot be None or empty")
        if 'embedding' not in X_train.columns:
            raise ValueError("Training data X must contain 'embedding' column")
        
        # possible edge case...
        # Ensure embeddings are in correct format
        for emb in X_train['embedding']:
            if not isinstance(emb, (list, np.ndarray)):
                raise ValueError("Each embedding must be a list or numpy array")
            if len(emb) == 0:
                raise ValueError("Embeddings cannot be empty")
        # Ensure embeddings have consistent shape
        emb_shape = X_train['embedding'].iloc[0].shape
        for emb in X_train['embedding']:
            if emb.shape != emb_shape:
                raise ValueError("All embeddings must have the same shape")
        # Check for null values            
        if X_train['embedding'].isnull().any():
            raise ValueError("Embeddings cannot contain null values")

    def _process_single_face(self, image, bbox, landmarks, scores, face_index: int):
        """Process a single detected face through the pipeline."""
        try:
            # Crop face
            face = Preprocessor.crop(image, bbox)
            if face is None:
                self.logger.warning(f"Failed to crop face {face_index}")
                return None, None
            
            # Generate embedding if embedder is available
            embedding = None
            if self.embedder is not None:
                try:
                    embedding = self.embedder.embed_face(face)
                except Exception as e:
                    self.logger.warning(f"Failed to generate embedding for face {face_index}: {e}")
            
            # Prepare face result
            face_result = {
                "face_id": face_index,
                "bbox": bbox,
                "landmarks": landmarks,
                "score": scores,
                "embedding": embedding,
                "label": None
            }
            
            # Classify face if classifier and embedding are available
            if self.classifier is not None and embedding is not None:
                try:
                    label = self.classifier.predict(embedding)
                    face_result["label"] = label
                except Exception as e:
                    self.logger.warning(f"Failed to classify face {face_index}: {e}")
                    face_result["label"] = None
            
            return face, face_result
            
        except Exception as e:
            self.logger.error(f"Error processing face {face_index}: {e}")
            return None, None

    