import numpy as np
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from lib.API.Reporter import Reporter, ReporterConfig
from lib.API.Pipeline import Pipeline, PipelineTask
from lib.API.Preprocessor import Preprocessor

# Test configuration
TEST_EMBEDDING_SIZE = 512
TEST_BBOX = (10, 20, 50, 60)
TEST_LANDMARKS = [[30, 40], [60, 40], [45, 55], [35, 70], [55, 70]]
TEST_SCORE = 0.95


@pytest.mark.unit
class TestPipelineComponentValidation:
    """Unit tests for pipeline component validation."""
    
    def test_validate_components_for_processing_missing_detector(self, reporter):
        """Test validation fails when detector is missing."""
        pipeline = Pipeline(reporter=reporter, detector=None)
        
        with pytest.raises(ValueError, match="FaceDetector is required"):
            pipeline._validate_components_for_processing()
    
    def test_validate_components_for_processing_missing_reporter(self, mock_detector):
        """Test validation fails when reporter is missing."""
        pipeline = Pipeline(reporter=None, detector=mock_detector)
        
        with pytest.raises(ValueError, match="Reporter is required"):
            pipeline._validate_components_for_processing()
    
    def test_validate_components_for_training_missing_reporter(self, mock_detector, mock_embedder, mock_classifier):
        """Test validation fails when reporter is missing for training."""
        pipeline = Pipeline(reporter=None, detector=mock_detector, embedder=mock_embedder, classifier=mock_classifier)
        
        with pytest.raises(ValueError, match="Reporter is required"):
            pipeline._validate_components_for_training()
    
    def test_validate_pipeline_for_task_success(self, pipeline_full: Pipeline):
        """Test task validation succeeds for valid pipeline."""
        assert pipeline_full.validate_pipeline_for_task(PipelineTask.PROCESS) == True
        assert pipeline_full.validate_pipeline_for_task(PipelineTask.TRAIN) == True
    
    def test_validate_pipeline_for_task_failure(self, reporter):
        """Test task validation fails for incomplete pipeline."""
        pipeline = Pipeline(reporter=reporter, detector=None)
        assert pipeline.validate_pipeline_for_task(PipelineTask.PROCESS) == False


@pytest.mark.unit
class TestPipelineDataValidation:
    """Unit tests for pipeline data validation."""
    
    def test_validate_train_data_empty_dataframe(self, pipeline_full: Pipeline):
        """Test training data validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Training data X cannot be None or empty"):
            pipeline_full._validate_train_data(empty_df)
    
    def test_validate_train_data_missing_embedding_column(self, pipeline_full: Pipeline):
        """Test training data validation with missing embedding column."""
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Training data X must contain 'embedding' column"):
            pipeline_full._validate_train_data(df)
    
    def test_validate_train_data_invalid_embedding_type(self, pipeline_full: Pipeline):
        """Test training data validation with invalid embedding type."""
        df = pd.DataFrame(
            {"embedding": ["not_an_array", "also_not_an_array"]},
        )
        
        with pytest.raises(ValueError, match="Each embedding must be a list or numpy array"):
            pipeline_full._validate_train_data(df)
    
    def test_validate_train_data_inconsistent_embedding_shape(self, pipeline_full: Pipeline):
        """Test training data validation with inconsistent embedding shapes."""
        df = pd.DataFrame({
            "embedding": [
                np.random.rand(128),
                np.random.rand(64)  # Different shape
            ]
        })
        
        with pytest.raises(ValueError, match="All embeddings must have the same shape"):
            pipeline_full._validate_train_data(df)
    
    def test_validate_train_data_null_embeddings(self, pipeline_full):
        """Test training data validation with null embeddings."""
        df = pd.DataFrame({"embedding": [np.random.rand(512), None]})
        
        with pytest.raises(ValueError, match="Embeddings cannot contain null values"):
            pipeline_full._validate_train_data(df)
    
    def test_validate_train_data_valid(self, pipeline_full: Pipeline):
        """Test training data validation with valid data."""
        df = pd.DataFrame({
            "embedding": [np.random.rand(512), np.random.rand(512)]
        })
        
        # Should not raise any exception
        pipeline_full._validate_train_data(df)


@pytest.mark.unit
class TestPipelineImageProcessing:
    """Unit tests for image processing methods."""
    
    def test_load_and_validate_image_file_not_exists(self, pipeline_minimal: Pipeline):
        """Test loading non-existent image file."""
        non_existent_path = Path("non_existent_image.jpg")
        
        with pytest.raises(FileNotFoundError):
            pipeline_minimal._load_and_validate_image(non_existent_path)

    def test_load_and_validate_image_invalid_format(self, pipeline_minimal: Pipeline, tmp_path):
        """Test loading invalid image format."""
        invalid_file: Path = tmp_path / "invalid.txt"
        invalid_file.write_text("not an image")
        
        with pytest.raises(ValueError, match="Error loading image"):
            pipeline_minimal._load_and_validate_image(invalid_file)
    
    @patch('lib.API.Preprocessor.Preprocessor.load')
    def test_load_and_validate_image_success(self, mock_load: Mock, pipeline_minimal: Pipeline, dummy_image_path):
        """Test successful image loading."""
        mock_load.return_value = np.random.rand(100, 100, 3)
        
        result = pipeline_minimal._load_and_validate_image(dummy_image_path)
        assert result is not None
        mock_load.assert_called_once_with(dummy_image_path)

    def test_process_single_face_success(self, pipeline_full: Pipeline, dummy_image_array):
        """Test successful single face processing."""
        face, result = pipeline_full._process_single_face(
            image=dummy_image_array,
            bbox=TEST_BBOX,
            landmarks=TEST_LANDMARKS,
            score=TEST_SCORE,
            face_index=0
        )
        
        assert face is not None
        assert result is not None
        assert result["face_id"] == 0
        assert result["bbox"] == TEST_BBOX
        assert result["landmarks"] == TEST_LANDMARKS
        assert result["score"] == TEST_SCORE
        assert "embedding" in result
        assert result["embedding"] is not None
    
    def test_process_single_face_crop_failure(self, pipeline_full: Pipeline, dummy_image_array):
        """Test single face processing with crop failure."""
        # Mock crop to return None
        with patch('lib.API.Preprocessor.Preprocessor.crop', return_value=None):
            face, result = pipeline_full._process_single_face(
                image=dummy_image_array,
                bbox=TEST_BBOX,
                landmarks=TEST_LANDMARKS,
                score=TEST_SCORE,
                face_index=0
            )
            
            assert face is None
            assert result is None

    def test_process_single_face_embedding_failure(self, pipeline_full: Pipeline, dummy_image_array):
        """Test single face processing with embedding failure."""
        # Mock embedder to raise exception
        pipeline_full.embedder.embed_face.side_effect = Exception("Embedding failed")
        
        face, result = pipeline_full._process_single_face(
            image=dummy_image_array,
            bbox=TEST_BBOX,
            landmarks=TEST_LANDMARKS,
            score=TEST_SCORE,
            face_index=0
        )
        
        assert face is not None
        assert result is not None
        assert result["embedding"] is None  # Should be None due to embedding failure


@pytest.mark.unit
class TestPipelineProcessMethod:
    """Unit tests for the process method."""
    
    def test_process_no_faces_detected(self, pipeline_full, dummy_image_path):
        """Test processing image with no faces detected."""
        # Mock detector to return no faces
        pipeline_full.detector.detect_faces.return_value = ([], [], [])
        
        result = pipeline_full.process(dummy_image_path)
        
        assert result["success"] == True
        assert result["faces_detected"] == 0
        assert result["faces_processed"] == 0
        assert len(result["errors"]) == 0

    def test_process_with_faces_detected(self, pipeline_full: Pipeline, dummy_image_path: Path):
        """Test processing image with faces detected."""
        # Mock detector to return one face
        pipeline_full.detector.detect_faces.return_value = ([TEST_BBOX], [TEST_LANDMARKS], [TEST_SCORE])
        
        result = pipeline_full.process(dummy_image_path)
        
        assert result["success"] == True
        assert result["faces_detected"] == 1
        assert result["faces_processed"] == 1
        assert len(result["errors"]) == 0
        
        # Verify reporter.save was called
        pipeline_full.reporter.save_processed_image_results.assert_called_once()
    
    def test_process_partial_face_processing_failure(self, pipeline_full, dummy_image_path):
        """Test processing with some faces failing to process."""
        # Mock detector to return two faces
        pipeline_full.detector.detect_faces.return_value = ([TEST_BBOX, TEST_BBOX], [TEST_LANDMARKS, TEST_LANDMARKS], [TEST_SCORE, TEST_SCORE])
        
        # Mock crop to fail for second face
        with patch('lib.API.Preprocessor.Preprocessor.crop') as mock_crop:
            mock_crop.side_effect = [np.random.rand(50, 50, 3), None]  # First succeeds, second fails
            
            result = pipeline_full.process(dummy_image_path)
            
            assert result["success"] == True
            assert result["faces_detected"] == 2
            assert result["faces_processed"] == 1  # Only one processed successfully
            assert len(result["errors"]) == 1

    def test_process_detector_failure(self, pipeline_full: Pipeline, dummy_image_path: Path):
        """Test processing with detector failure."""
        pipeline_full.detector.detect_faces.side_effect = Exception("Detector failed")
        
        result = pipeline_full.process(dummy_image_path)
        
        assert result["success"] == False
        assert len(result["errors"]) == 1
        assert "Detector failed" in result["errors"][0]


@pytest.mark.unit
class TestPipelineBulkProcessMethod:
    """Unit tests for the bulk_process method."""
    
    def test_bulk_process_empty_image_list(self, pipeline_full):
        """Test bulk processing with empty image list."""
        with pytest.raises(ValueError, match="image_paths cannot be empty"):
            pipeline_full.bulk_process([])
    
    def test_bulk_process_all_success(self, pipeline_full, multiple_dummy_images):
        """Test bulk processing with all images successful."""
        # Mock detector to return one face for each image
        pipeline_full.detector.detect_faces.return_value = ([TEST_BBOX], [TEST_LANDMARKS], [TEST_SCORE])
        
        result = pipeline_full.bulk_process(multiple_dummy_images)
        
        assert result["success"] == True
        assert result["total_images"] == len(multiple_dummy_images)
        assert result["processed_images"] == len(multiple_dummy_images)
        assert result["failed_images"] == 0
        assert len(result["errors"]) == 0
    
    def test_bulk_process_some_failures_continue_on_error(self, pipeline_full, multiple_dummy_images):
        """Test bulk processing with some failures, continuing on error."""
        # Mock detector to fail on second image
        call_count = 0
        def mock_detect_faces(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Detector failed")
            return ([TEST_BBOX], [TEST_LANDMARKS], [TEST_SCORE])
        
        pipeline_full.detector.detect_faces.side_effect = mock_detect_faces
        
        result = pipeline_full.bulk_process(multiple_dummy_images, continue_on_error=True)
        
        assert result["success"] == True
        assert result["processed_images"] == len(multiple_dummy_images) - 1
        assert result["failed_images"] == 1
        assert len(result["errors"]) > 0
    
    def test_bulk_process_stop_on_error(self, pipeline_full, multiple_dummy_images):
        """Test bulk processing stopping on first error."""
        # Mock detector to fail on first image
        pipeline_full.detector.detect_faces.side_effect = Exception("Detector failed")
        
        result = pipeline_full.bulk_process(multiple_dummy_images, continue_on_error=False)
        
        assert result["success"] == False
        assert result["processed_images"] == 0
        assert result["failed_images"] == 1
    
    def test_bulk_process_progress_callback(self, pipeline_full, multiple_dummy_images):
        """Test bulk processing with progress callback."""
        progress_calls = []
        
        def progress_callback(current, total, path):
            progress_calls.append((current, total, path))
        
        # Mock detector
        pipeline_full.detector.detect_faces.return_value = ([TEST_BBOX], [TEST_LANDMARKS], [TEST_SCORE])
        
        pipeline_full.bulk_process(multiple_dummy_images, progress_callback=progress_callback)
        
        assert len(progress_calls) == len(multiple_dummy_images)
        assert progress_calls[0] == (1, len(multiple_dummy_images), multiple_dummy_images[0])
        assert progress_calls[-1] == (len(multiple_dummy_images), len(multiple_dummy_images), multiple_dummy_images[-1])


@pytest.mark.unit
class TestPipelineTrainMethod:
    """Unit tests for the train method."""
    
    def test_train_success_with_auto_clusters(self, pipeline_full):
        """Test successful training with automatic cluster estimation."""
        # Create valid training data
        embeddings = [np.random.rand(TEST_EMBEDDING_SIZE) for _ in range(10)]
        train_df = pd.DataFrame({"embedding": embeddings})
        
        result = pipeline_full.train(train_df, max_clusters=5)
        
        assert result["success"] == True
        assert result["n_clusters_found"] is not None
        assert result["train_embedding__samples"] == 10
        assert result["silhouette_scores"] is not None
        assert result["inertia"] is not None
    
    def test_train_insufficient_data_for_clustering(self, pipeline_full):
        """Test training with insufficient data for clustering."""
        # Single embedding - cannot form clusters
        train_df = pd.DataFrame({"embedding": [np.random.rand(TEST_EMBEDDING_SIZE)]})
        
        result = pipeline_full.train(train_df)
        
        # Should handle gracefully and create single cluster
        assert result["success"] == True
        assert result["n_clusters_found"] == 1
    
    def test_train_clustering_failure(self, pipeline_full):
        """Test training with clustering failure."""
        embeddings = [np.random.rand(TEST_EMBEDDING_SIZE) for _ in range(10)]
        train_df = pd.DataFrame({"embedding": embeddings})
        
        # Mock clustering to fail
        with patch('sklearn.cluster.KMeans') as mock_kmeans:
            mock_kmeans.side_effect = Exception("Clustering failed")
            
            result = pipeline_full.train(train_df)
            
            assert result["success"] == False
            assert "Clustering failed" in str(result["errors"])


@pytest.mark.integration
@pytest.mark.real_models
class TestPipelineIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_single_image_processing(self, pipeline_full_real: Pipeline, dummy_image_path: Path):
        """Test complete pipeline from image to results."""
        result = pipeline_full_real.process(dummy_image_path)
        
        # Basic success validation
        assert isinstance(result, dict)
        assert "success" in result
        assert "faces_detected" in result
        assert "faces_processed" in result
        assert isinstance(result["faces_detected"], int)
        assert isinstance(result["faces_processed"], int)
        assert result["faces_processed"] <= result["faces_detected"]
    
    def test_end_to_end_training_and_inference(self, pipeline_full_real: Pipeline):
        """Test training followed by inference."""
        # Generate synthetic training data
        embeddings = [np.random.rand(TEST_EMBEDDING_SIZE) for _ in range(20)]
        train_df = pd.DataFrame({"embedding": embeddings})
        
        # Train the classifier
        train_result = pipeline_full_real.train(train_df)
        assert train_result["success"] == True
        
        # Test that classifier was updated
        assert pipeline_full_real.classifier is not None
    
    def test_pipeline_with_different_component_combinations(self, reporter):
        """Test pipeline with various component combinations."""
        detector = Mock()
        detector.detect_faces.return_value = ([TEST_BBOX], [TEST_LANDMARKS], [TEST_SCORE])
        
        # Test detector only
        pipeline_detect_only = Pipeline(reporter=reporter, detector=detector)
        assert pipeline_detect_only.validate_pipeline_for_task(PipelineTask.PROCESS) == True
        assert pipeline_detect_only.validate_pipeline_for_task(PipelineTask.TRAIN) == False
        
        # Test detector + embedder
        embedder = Mock()
        embedder.embed_face.return_value = np.random.rand(TEST_EMBEDDING_SIZE)
        pipeline_detect_embed = Pipeline(reporter=reporter, detector=detector, embedder=embedder)
        assert pipeline_detect_embed.validate_pipeline_for_task(PipelineTask.PROCESS) == True
        assert pipeline_detect_embed.validate_pipeline_for_task(PipelineTask.TRAIN) == False


# Additional fixtures for pipeline tests
@pytest.fixture
def mock_detector() -> Mock:
    """Create a mock detector."""
    detector = Mock()
    detector.detect_faces.return_value = ([TEST_BBOX], [TEST_LANDMARKS], [TEST_SCORE])
    detector.settings.return_value = {"detector": "mock"}
    return detector


@pytest.fixture
def mock_embedder() -> Mock:
    """Create a mock embedder."""
    embedder = Mock()
    embedder.embed_face.return_value = np.random.rand(TEST_EMBEDDING_SIZE)
    embedder.settings.return_value = {"embedder": "mock"}
    return embedder


@pytest.fixture
def mock_classifier() -> Mock:
    """Create a mock classifier."""
    classifier = Mock()
    classifier.fit.return_value = None
    classifier.predict.return_value = [0, 1, 0]
    classifier.settings.return_value = {"classifier": "mock"}
    return classifier


@pytest.fixture
def mock_reporter() -> Mock:
    """Create a mock reporter."""
    reporter = Mock()
    reporter.config = Mock()
    reporter.config.is_saving_enabled = True
    reporter.setup_output_directory.return_value = None
    reporter.setup_bulk_output_directory.return_value = None
    reporter.setup_training_output_directory.return_value = None
    reporter.save_processed_image_results.return_value = None
    reporter.save_test_summary.return_value = None
    return reporter


@pytest.fixture
def pipeline_minimal(mock_reporter, mock_detector):
    """Create a minimal pipeline for testing."""
    return Pipeline(reporter=mock_reporter, detector=mock_detector)


@pytest.fixture
def pipeline_full(mock_reporter, mock_detector, mock_embedder, mock_classifier):
    """Create a full pipeline with all components for testing."""
    return Pipeline(
        reporter=mock_reporter,
        detector=mock_detector,
        embedder=mock_embedder,
        classifier=mock_classifier
    )


@pytest.fixture
def pipeline_full_real(reporter):
    """Create a pipeline with real components for integration testing."""
    # Import actual implementations
    from lib.face_detection.YuNet import YuNetDetector
    from lib.face_representation.ArcFace import ArcFaceEmbedder
    from lib.face_classification.KMeansClassifier import KMeansClassifier
    from sklearn.cluster import KMeans
    detector = YuNetDetector()
    embedder = ArcFaceEmbedder()
    classifier = KMeansClassifier(KMeans())
    
    return Pipeline(
        reporter=reporter,
        detector=detector,
        embedder=embedder,
        classifier=classifier
    )


@pytest.fixture
def dummy_image_array():
    """Create a dummy image array."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def dummy_image_path(tmp_path, dummy_image_array):
    """Create a dummy image file."""
    import cv2
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), dummy_image_array)
    return image_path


@pytest.fixture
def multiple_dummy_images(tmp_path, dummy_image_array):
    """Create multiple dummy image files."""
    import cv2
    images = []
    for i in range(3):
        image_path = tmp_path / f"test_image_{i}.jpg"
        cv2.imwrite(str(image_path), dummy_image_array)
        images.append(image_path)
    return images
