from pathlib import Path
import cv2
import numpy as np

from lib.face_classification.LoadedClassifier import LoadedClassifier
from lib.face_classification.KMeansClassifier import KMeansClassifier
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
from sklearn.cluster import KMeans , MiniBatchKMeans, DBSCAN , AgglomerativeClustering , OPTICS , Birch
# https://www.geeksforgeeks.org/machine-learning/what-is-silhouette-score/
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


class PipelineTask(Enum):
    BULK_PROCESS = "bulk_process"
    PROCESS = "process"
    INFERENCE = "inference"
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
# where i have a detector (RetinaFace,Viola Jones, YuNet), an embedder/represent (ArcFace) and optionally a classifier (e.g. Knn, DBSCAN, etc.)
# the only thing that i add is the automatic reporting/saving of results which i believe is very useful for practical applications
# and where larger datasets are used, the reporting becomes essential

# the pipeline should be able to handle single images as well as batches of images
# therefore Pipeline offers a bulk_process() method for batch processing and a process() method for single image processing
# and a train() method for unsupervised training using clustering on face embeddings
# the pipeline should also be able to handle errors gracefully and continue processing other images 
# which means that FaceDetector, FaceEmbedder and FaceClassifier are not responsible for error handling !!

# from a design perspective, the Pipeline has 4 main methods:
# - train(): for unsupervised training of the classifier using clustering on face embeddings
# - bulk_process(): for batch processing of images
# - process(): for single image processing
# - _process_single_face(): for processing a single detected face (align, embed, classify)
# each of these methods build upon the previous one
# however the user can also call process() directly for single image processing

class Pipeline:
    logger = logging.getLogger(__name__)

    def __init__(
        self, 
        reporter: Reporter,
        detector: FaceDetector, 
        embedder: FaceEmbedder = None,
        classifier: FaceClassifier = None,
    ):
        """Initialize Pipeline with components.
        
        Args:
            reporter: Reporter instance (use Pipeline.create_reporter() for easy setup)
            detector: Face detection model
            embedder: Face embedding model (optional)
            classifier: Face classification model (optional)
        """
        self.detector = detector
        self.embedder = embedder
        self.classifier = classifier
        self.reporter = reporter
        self.task_requirements = {
            PipelineTask.PROCESS: self._validate_components_for_processing,
            PipelineTask.BULK_PROCESS: self._validate_components_for_bulk_processing,
            PipelineTask.INFERENCE: self._validate_components_for_inference,
            PipelineTask.TRAIN: self._validate_components_for_training,
        }


    # -------------------------------------- Static methods for easy setup (less boilerplate) -------------------------------------- #

    @classmethod
    def create_reporter(
        cls,
        output_dir: Path,
        save_cropped_faces: bool = True,
        save_model_settings: bool = True,
        save_model: bool = True,
    ) -> Reporter:
        """Create a Reporter with simplified configuration.
        
        Args:
            output_dir: Base directory for saving outputs
            save_cropped_faces: Save cropped face images
            save_model_settings: Save model configuration
            save_model: Save trained models
            
        Returns:
            Configured Reporter instance
        """
        
        config = ReporterConfig(
            output_dir=output_dir,
            save_annotated_image=True,
            save_cropped_faces=save_cropped_faces,
            
            save_model=save_model,
            save_model_format=ModelFormat.ONNX,

            save_model_settings=save_model_settings,
            save_model_settings_format=OutputFormat.JSON,

            save_image_results_to_file=True,
            save_image_results_to_file_format=OutputFormat.JSON,
            
            save_compiled_results=True,
            save_compiled_results_format=OutputFormat.PARQUET,

            save_cosine_similarity_matrix=True,
            save_cosine_similarity_matrix_visualization=True,

            save_tsne_visualization=True,
        )
        
        return Reporter(config)

    def validate_pipeline_for_task(self, task: str) -> bool:
        """
        Validate that the pipeline has the required components for a specific task.
        
        Args:
            task: Task type ('detect', 'embed', 'classify', 'train')
            
        Returns:
            bool: True if pipeline can perform the task
        """
        try:
            task_enum = PipelineTask(task)
            self.task_requirements[task_enum]()
            return True
        except (ValueError, KeyError , RuntimeError) as e:
            self.logger.error(f"Pipeline validation failed for task '{task}': {e}")
            return False
        
    # -------------------------------------- Core pipeline processing methods -------------------------------------- #
    def inference(
        self, 
    ):
        """
        The inference should be a optimized method for identifying faces in real-time
        NOTE: currently not implemented
        """
        self.validate_pipeline_for_task(PipelineTask.INFERENCE)
        try:
            pass
        except Exception as e:
            raise RuntimeError(f"Inference error: {str(e)}")
        return None
    
    def process(self, image_path: Path) -> dict:
        """
        Process a single image through the face detection/embedding pipeline.
        
        Args:
            image_path: Path to the image file to process
            
        Returns:
            dict: Processing results containing faces found, errors, and metadata
        """
        # Pipeline steps:
        # ---------------
        # 1. Load and validate image
        # 2. Detect faces
        # 3. Process each detected face
        #   3.1 align face or crop face
        #   3.2 embed face (if embedder available)
        # 4. Save results
        
        # Validate components and inputs
        self.validate_pipeline_for_task(PipelineTask.PROCESS)
        
        processing_result = {
            "image_path": image_path.as_posix(),
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
            processing_result["faces_detected"] = len(bboxes) if bboxes is not None else 0
            # if no faces were detected (bboxes == 0) we can conclude 2 things:
                # - either the image is valid but no faces are present
                # - or the image is still valid but the faces are too small or occluded for the detector to find them
            # how to distinguish between these 2 cases ?
            # well we can't really do anything on the fly
            # however what we can do is try to adjust the detector settings (e.g. lower the confidence threshold, increase the input size, etc.)
            # and re-run the detection (currently not implemented).
            # but even then it's not a guarantee that no faces are present
            # that is why reporting is important, so the user can review the results and decide if the detector is working well enough for their use case
            if processing_result["faces_detected"] == 0:
                self.logger.info(f"No faces detected in image: {image_path}")
                processing_result["success"] = True
                return processing_result

            self.logger.info(f"Detected {len(bboxes)} faces in image: {image_path}")
            
            # 3) process each detected face
            results = []
            cropped_faces = []

            for (i, bbox), landmark, score in zip(enumerate(bboxes), landmarks, scores):
                # 3.1-4) align, crop, embed and collect results
                face, face_result = self._process_single_face(image, bbox, landmark, score, i)

                if face is not None and face_result is not None:
                    cropped_faces.append(face)
                    results.append(face_result)
                    processing_result["faces_processed"] += 1
                else:
                    processing_result["errors"].append(f"Failed to process face {i}")
            
            # 4) save results if any faces were successfully processed
            if results:
                self.logger.debug(f"Saving results for {len(results)} faces from image: {image_path}")
                self.reporter.save_processed_image_results(
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

    def bulk_process(
        self, 
        image_paths: list[Path], 
        continue_on_error: bool = True, 
        progress_callback=None,
    ):
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

        self.validate_pipeline_for_task(PipelineTask.BULK_PROCESS)

        self.reporter.setup_bulk_output_directory()

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
            "succeeded_paths": [],
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
                processing_result = self.process(image_path)

                # 6) Aggregate results
                if processing_result["success"]:
                    batch_result["processed_images"] += 1
                    batch_result["total_faces_detected"] += processing_result["faces_detected"]
                    batch_result["total_faces_processed"] += processing_result["faces_processed"]
                    batch_result["succeeded_paths"].append(image_path.as_posix())
                else:
                    batch_result["failed_images"] += 1
                    batch_result["failed_paths"].append(image_path.as_posix())
                    batch_result["errors"].extend(processing_result["errors"])
                    
                    if not continue_on_error:
                        batch_result["success"] = False
                        self.logger.error(f"Stopping batch processing due to error in {image_path}")
                        return
                        
            except Exception as e:
                error_msg = f"Unexpected error processing {image_path}: {str(e)}"
                self.logger.error(error_msg)
                batch_result["failed_images"] += 1
                batch_result["failed_paths"].append(image_path.as_posix())
                batch_result["errors"].append(error_msg)
                
                if not continue_on_error:
                    batch_result["success"] = False
                    return

        # 5) save results
        self.logger.info(
            f"Batch processing completed: "
            f"{batch_result['processed_images']}/{batch_result['total_images']} images successful, "
            f"{batch_result['total_faces_processed']} faces processed"
            f" with {batch_result['failed_images']} failures"
        )
        self.reporter.save_batch_summary(batch_result)

    def train(
        self, 
        X: DataFrame, 
        max_clusters: int = 20,
        random_state: int = 42
    ):
        """
        Train the classifier using unsupervised clustering on face embeddings.
        Face embeddings come in as a DataFrame.
        Only 1 classifier is supported for now:
            - KMeans \n https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        number of clusters can be specified or estimated automatically using the elbow method and silhouette analysis \n
        https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

        Args:
            X: DataFrame with 'embedding' column containing face embeddings
            max_clusters: Maximum number of clusters to consider when estimating optimal clusters
            random_state: Random seed for reproducible results
        """
        # pipeline steps:
        # ---------------
        # 1. Validate components and inputs
        # 2. Determine optimal number of clusters
        # 3. Save training results and model

        # 1) Validate components and inputs
        self.validate_pipeline_for_task(PipelineTask.TRAIN)
        self._validate_train_data(X)

        X_TRAIN = np.stack(X["embedding"].values)
        self.logger.info(f"Starting unsupervised training pipeline with {len(X_TRAIN)} training samples")

        # initialize training results
        # cluster centers are required for classification during inference
        # they are saved by the reporter in a separate file (npy and parquet)
        # so they can be loaded during inference.
        training_result = {
            "clustering_method": None,
            "n_clusters_found": None,
            "train_embedding__samples": len(X_TRAIN),
            "silhouette_scores": None,
            "inertia": None,
            "cluster_labels": None,
            "errors": [],
            "success": False,
        }
        
        classifier = None
        silhouette_scores = []
        inertias = []
        try:
            # 2) Determine optimal number of clusters by elbow method and silhouette analysis
            classifier, n_clusters, inertias , silhouette_scores = self._estimate_optimal_clusters(
                X_TRAIN, 
                max_clusters=min(max_clusters, len(X_TRAIN)//2), 
                random_state=random_state
            )
            training_result["inertia"] = inertias
            training_result["silhouette_scores"] = silhouette_scores
            training_result["n_clusters_found"] = n_clusters
            training_result["success"] = True
            self.logger.info(f"Estimated optimal number of clusters: {n_clusters}")
            self.logger.info(f"Unsupervised training completed successfully!")

        except Exception as e:
            error_msg = f"Unsupervised training pipeline failed: {e}"
            self.logger.error(error_msg)
            training_result["errors"].append(error_msg)
            training_result["success"] = False
        
        finally:
            # 3) Save training results and model
            self.reporter.save_test_summary(
                training_result,
                classifier=classifier,
                train_data=X_TRAIN,
                silhouette_scores=silhouette_scores,
                inertias=inertias,
            )
        return self

    # -------------------------------------- Helper methods for internal processing -------------------------------------- # 

    # -------------------------------- Clustering helpers -------------------------------------- #
    def _estimate_optimal_clusters(self, embeddings: pd.Series, max_clusters: int = 20, random_state: int = 42) -> tuple[KMeansClassifier,int, list[float], list[float]]:
        """Estimate optimal number of clusters using elbow method and silhouette analysis."""        
        if len(embeddings) < 2:
            raise ValueError("Not enough embeddings to estimate clusters")
        if len(embeddings) < max_clusters:
            raise ValueError("more clusters than samples")
            
        # we try different cluster numbers and evaluate the silhouette score
        silhouette_scores = []
        inertias = []
        K_range = range(2, max_clusters + 1)
        kmeans = None
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            silhouette_scores.append(silhouette_score(embeddings, cluster_labels,metric="euclidean"))
            inertias.append(kmeans.inertia_)
        
        # find elbow point by looking for max silhouette score (closest to 1 is better)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
        kmeans.fit_predict(embeddings)
        return (KMeansClassifier(kmeans), optimal_k , inertias, silhouette_scores)
        
    # -------------------------------- Pipeline processing helpers -------------------------------------- #
    def _validate_components_for_inference(self):
        """Validate that required (detector , embedder, classifier) components are available for inference."""
        if self.detector is None:
            raise ValueError("FaceDetector is required for the inference pipeline")
        if self.embedder is None:
            raise ValueError("FaceEmbedder is required for the inference pipeline")
        if self.classifier is None:
            raise ValueError("FaceClassifier is required for the inference pipeline")

    def _validate_components_for_bulk_processing(self):
        """Validate that required (embedder and classifier are optional) components are available for bulk processing."""
        if self.detector is None:
            raise ValueError("FaceDetector is required for the bulk processing pipeline")
        if self.reporter is None:
            raise ValueError("Reporter is required for the bulk processing pipeline")
        
    def _validate_components_for_processing(self):
        """Validate that required (embedder and classifier are optional) components are available for processing."""
        if self.detector is None:
            raise ValueError("FaceDetector is required for the processing pipeline")
        if self.reporter is None:
            raise ValueError("Reporter is required for the processing pipeline")

    def _validate_components_for_training(self):
        """Validate that required components are available for training."""
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
        emb_shape = self.embedder.output_shape
        for emb in X_train['embedding']:
            if emb.shape != emb_shape:
                raise ValueError("All embeddings must have the same shape")
        # Check for null values            
        if X_train['embedding'].isnull().any():
            raise ValueError("Embeddings cannot contain null values")

    def _process_single_face(
        self, 
        image,
        bbox,
        landmarks,
        score,
        face_index: int,
    ):
        """Process a single detected face through the pipeline."""
        try:
            aligned = False
            face = None
            # face - alignment or cropping
            if landmarks is not None and len(landmarks) == 5:
                try: 
                    # NOTE: align uses a strict 5-point landmark format (5,2) (mouth corners, nose tip, eye centers)
                    # NOTE: align uses Uniface's method "face_alignment" which has a "reference_alignment" value hardcoded
                    # this means the alignment will always be based on this standard reference if i/you would like to change this
                    # just copy the source code of face_alignment and implement your own version of "face_alignment"
                    face , _ = Preprocessor.align(image, landmarks)
                    aligned = True
                except ValueError as e:
                    # if alignment fails, we fallback to cropping with our bounding box
                    face = Preprocessor.crop(image, bbox)
                    if face is None:
                        # if cropping fails, we return because we can't proceed
                        # we can't embed or classify without a valid face image
                        self.logger.warning(f"Failed to crop face {face_index}")
                        return None, None
                    
            # face - embedding/representation
            # if a model is provided, we embed the face
            label = None
            embedding = None
            if self.embedder is not None:
                try:
                    embedding = self.embedder.embed_face(face)
                except Exception as e:
                    self.logger.warning(f"Failed to generate embedding for face {face_index}: {e}")
                
                # if embedding succeeded, we classify the face if a classifier is provided
                if self.classifier and embedding is not None:
                    try:
                        label = self.classifier.predict(embedding)
                        if label == -1:
                            label = "unknown"
                    except Exception as e:
                        self.logger.warning(f"Failed to generate label for face {face_index}: {e}")

            
            face_result = {
                "face_id": face_index,
                "bbox": bbox,
                "landmarks": landmarks,
                "score": score,
                "embedding": embedding,
                "aligned": aligned,
                "label": label,
            }
            return face, face_result
            
        except Exception as e:
            self.logger.error(f"Error processing face {face_index}: {e}")
            return None, None

    