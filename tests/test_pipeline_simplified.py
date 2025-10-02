"""
Simplified test suite for the Pipeline class focusing on main user functions.
Tests the core functionality: process(), bulk_process(), and train() methods
with various configuration scenarios and input types.
"""

import pytest
import numpy as np
import pandas as pd
import cv2
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


@pytest.mark.pipeline
@pytest.mark.integration
class TestPipelineProcess:
    """Test the main process() method with various scenarios."""
    
    def test_process_successful_face_detection(self, pipeline_full: Pipeline, dummy_image_path: Path):
        """Test successful single image processing with face detection."""
        # Mock detector to return one face
        pipeline_full.detector.detect_faces.return_value = ([TEST_BBOX], [TEST_LANDMARKS], [TEST_SCORE])
        
        result, processed_results = pipeline_full.process(dummy_image_path)
        
        assert result["success"] == True
        assert result["faces_detected"] == 1
        assert result["faces_processed"] == 1
        assert len(result["errors"]) == 0
        assert len(processed_results) == 1
        
        # Verify the processed result structure
        face_result = processed_results[0]
        assert "face_id" in face_result
        assert "bbox" in face_result
        assert "embedding" in face_result
        assert face_result["bbox"] == TEST_BBOX
        
        # Verify reporter was called
        pipeline_full.reporter.save_processed_image_results.assert_called_once()
    
    def test_process_no_faces_detected(self, pipeline_full: Pipeline, dummy_image_path: Path):
        """Test processing image with no faces detected."""
        # Mock detector to return no faces
        pipeline_full.detector.detect_faces.return_value = ([], [], [])
        
        result, processed_results = pipeline_full.process(dummy_image_path)
        
        assert result["success"] == True
        assert result["faces_detected"] == 0
        assert result["faces_processed"] == 0
        assert len(processed_results) == 0
        assert len(result["errors"]) == 0
    
    def test_process_detector_only_pipeline(self, mock_reporter, mock_detector, dummy_image_path):
        """Test processing with detector-only pipeline (no embedder/classifier)."""
        pipeline = Pipeline(reporter=mock_reporter, detector=mock_detector)
        mock_detector.detect_faces.return_value = ([TEST_BBOX], [TEST_LANDMARKS], [TEST_SCORE])
        
        result, processed_results = pipeline.process(dummy_image_path)
        
        assert result["success"] == True
        assert result["faces_detected"] == 1
        assert len(processed_results) == 1
        # Should work without embedder/classifier
        face_result = processed_results[0]
        assert "bbox" in face_result
        assert face_result["embedding"] is None  # No embedder
    
    def test_process_invalid_image_file(self, pipeline_full: Pipeline, tmp_path):
        """Test processing with non-existent image file."""
        non_existent_path = tmp_path / "non_existent.jpg"
        
        result, processed_results = pipeline_full.process(non_existent_path)
        
        assert result["success"] == False
        assert len(result["errors"]) > 0
        assert "FileNotFoundError" in str(result["errors"][0]) or "not found" in str(result["errors"][0]).lower()
    
    def test_process_detector_failure(self, pipeline_full: Pipeline, dummy_image_path: Path):
        """Test processing when detector fails."""
        pipeline_full.detector.detect_faces.side_effect = Exception("Detector failed")
        
        result, processed_results = pipeline_full.process(dummy_image_path)
        
        assert result["success"] == False
        assert len(result["errors"]) > 0
        assert "Detector failed" in str(result["errors"][0])


@pytest.mark.pipeline
@pytest.mark.integration
class TestPipelineBulkProcess:
    """Test the main bulk_process() method with various scenarios."""
    
    def test_bulk_process_all_successful(self, pipeline_full: Pipeline, multiple_dummy_images):
        """Test bulk processing with all images successful."""
        # Mock detector to return one face for each image
        pipeline_full.detector.detect_faces.return_value = ([TEST_BBOX], [TEST_LANDMARKS], [TEST_SCORE])
        
        result = pipeline_full.bulk_process(multiple_dummy_images)
        
        assert result["success"] == True
        assert result["total_images"] == len(multiple_dummy_images)
        assert result["processed_images"] == len(multiple_dummy_images)
        assert result["failed_images"] == 0
        assert result["total_faces_detected"] == len(multiple_dummy_images)  # One face per image
        assert result["total_faces_processed"] == len(multiple_dummy_images)
        assert len(result["errors"]) == 0
        
        # Verify bulk mode was set up
        pipeline_full.reporter.setup_bulk_output_directory.assert_called_once()
    
    def test_bulk_process_some_failures_continue_on_error(self, pipeline_full: Pipeline, multiple_dummy_images):
        """Test bulk processing with some failures, continuing on error."""
        call_count = 0
        def mock_detect_faces(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second image
                raise Exception("Detection failed")
            return ([TEST_BBOX], [TEST_LANDMARKS], [TEST_SCORE])
        
        pipeline_full.detector.detect_faces.side_effect = mock_detect_faces
        
        result = pipeline_full.bulk_process(multiple_dummy_images, continue_on_error=True)
        
        assert result["success"] == True  # Overall success despite one failure
        assert result["total_images"] == len(multiple_dummy_images)
        assert result["processed_images"] == len(multiple_dummy_images) - 1
        assert result["failed_images"] == 1
        assert len(result["errors"]) > 0
        assert len(result["failed_paths"]) == 1
        assert len(result["succeeded_paths"]) == len(multiple_dummy_images) - 1
    
    def test_bulk_process_stop_on_error(self, pipeline_full: Pipeline, multiple_dummy_images):
        """Test bulk processing stopping on first error."""
        # Mock detector to fail immediately
        pipeline_full.detector.detect_faces.side_effect = Exception("Detector failed")
        
        result = pipeline_full.bulk_process(multiple_dummy_images, continue_on_error=False)
        
        assert result["success"] == False
        assert result["processed_images"] == 0
        assert result["failed_images"] == 1
        assert len(result["errors"]) > 0
    
    def test_bulk_process_empty_image_list(self, pipeline_full: Pipeline):
        """Test bulk processing with empty image list."""
        with pytest.raises(ValueError, match="image_paths cannot be empty"):
            pipeline_full.bulk_process([])
    
    def test_bulk_process_progress_callback(self, pipeline_full: Pipeline, multiple_dummy_images):
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


@pytest.mark.pipeline
@pytest.mark.integration
class TestPipelineTrain:
    """Test the main train() method with various scenarios."""
    
    def test_train_successful_with_sufficient_data(self, pipeline_full: Pipeline):
        """Test successful training with sufficient embedding data."""
        # Create realistic training data
        embeddings = [np.random.rand(TEST_EMBEDDING_SIZE).astype(np.float32) for _ in range(20)]
        train_df = pd.DataFrame({"embedding": embeddings})
        
        result = pipeline_full.train(train_df, max_clusters=5)
        
        assert result["success"] == True
        assert result["n_clusters_found"] is not None
        assert result["n_clusters_found"] >= 1
        assert result["train_embedding__samples"] == 20
        assert result["silhouette_scores"] is not None
        assert result["inertia"] is not None
        assert len(result["errors"]) == 0
        
        # Verify reporter save was called
        pipeline_full.reporter.save_test_summary.assert_called_once()
    
    def test_train_minimal_data(self, pipeline_full: Pipeline):
        """Test training with minimal data (edge case)."""
        # Just enough data for clustering
        embeddings = [np.random.rand(TEST_EMBEDDING_SIZE).astype(np.float32) for _ in range(3)]
        train_df = pd.DataFrame({"embedding": embeddings})
        
        result = pipeline_full.train(train_df, max_clusters=2)
        
        assert result["success"] == True
        assert result["n_clusters_found"] is not None
        assert result["train_embedding__samples"] == 3
    
    def test_train_invalid_data_format(self, pipeline_full: Pipeline):
        """Test training with invalid data format."""
        # Invalid DataFrame without embedding column
        invalid_df = pd.DataFrame({"other_column": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Training data X must contain 'embedding' column"):
            pipeline_full.train(invalid_df)
    
    def test_train_empty_dataframe(self, pipeline_full: Pipeline):
        """Test training with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Training data X cannot be None or empty"):
            pipeline_full.train(empty_df)
    
    def test_train_inconsistent_embedding_shapes(self, pipeline_full: Pipeline):
        """Test training with inconsistent embedding shapes."""
        embeddings = [
            np.random.rand(512).astype(np.float32),
            np.random.rand(256).astype(np.float32)  # Different shape
        ]
        train_df = pd.DataFrame({"embedding": embeddings})
        
        with pytest.raises(ValueError, match="All embeddings must have the same shape"):
            pipeline_full.train(train_df)


@pytest.mark.pipeline
@pytest.mark.unit
class TestPipelineConfiguration:
    """Test pipeline configuration and validation scenarios."""
    
    def test_pipeline_task_validation_success(self, pipeline_full: Pipeline):
        """Test task validation for valid pipeline configurations."""
        assert pipeline_full.validate_pipeline_for_task(PipelineTask.PROCESS.value) == True
        assert pipeline_full.validate_pipeline_for_task(PipelineTask.BULK_PROCESS.value) == True
        assert pipeline_full.validate_pipeline_for_task(PipelineTask.TRAIN.value) == True
    
    def test_pipeline_task_validation_failures(self, mock_reporter):
        """Test task validation failures for incomplete pipelines."""
        # Pipeline with no detector
        pipeline_no_detector = Pipeline(reporter=mock_reporter, detector=None)
        assert pipeline_no_detector.validate_pipeline_for_task(PipelineTask.PROCESS.value) == False
        
        # Pipeline with no reporter
        mock_detector = Mock()
        pipeline_no_reporter = Pipeline(reporter=None, detector=mock_detector)
        assert pipeline_no_reporter.validate_pipeline_for_task(PipelineTask.PROCESS.value) == False
    
    def test_pipeline_component_combinations(self, mock_reporter):
        """Test different valid component combinations."""
        mock_detector = Mock()
        mock_embedder = Mock()
        mock_classifier = Mock()
        
        # Detector only
        pipeline_detect = Pipeline(reporter=mock_reporter, detector=mock_detector)
        assert pipeline_detect.validate_pipeline_for_task(PipelineTask.PROCESS.value) == True
        assert pipeline_detect.validate_pipeline_for_task(PipelineTask.TRAIN.value) == False
        
        # Detector + Embedder
        pipeline_detect_embed = Pipeline(
            reporter=mock_reporter, 
            detector=mock_detector, 
            embedder=mock_embedder
        )
        assert pipeline_detect_embed.validate_pipeline_for_task(PipelineTask.PROCESS.value) == True
        assert pipeline_detect_embed.validate_pipeline_for_task(PipelineTask.TRAIN.value) == False
        
        # Full pipeline
        pipeline_full = Pipeline(
            reporter=mock_reporter,
            detector=mock_detector,
            embedder=mock_embedder,
            classifier=mock_classifier
        )
        assert pipeline_full.validate_pipeline_for_task(PipelineTask.PROCESS.value) == True
        assert pipeline_full.validate_pipeline_for_task(PipelineTask.TRAIN.value) == True


@pytest.mark.pipeline
@pytest.mark.integration
@pytest.mark.slow
class TestPipelineRealModels:
    """Integration tests with real models (if available)."""
    
    def test_end_to_end_processing_real_models(self, pipeline_full_real: Pipeline, dummy_image_path: Path):
        """Test complete pipeline with real models."""
        try:
            result, processed_results = pipeline_full_real.process(dummy_image_path)
            
            # Basic validation - should not crash
            assert isinstance(result, dict)
            assert "success" in result
            assert "faces_detected" in result
            assert "faces_processed" in result
            assert isinstance(processed_results, list)
            
        except ImportError:
            pytest.skip("Real models not available - skipping integration test")
        except Exception as e:
            pytest.fail(f"Real model integration test failed: {e}")


# Simplified fixtures
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
    embedder.embed_face.return_value = np.random.rand(TEST_EMBEDDING_SIZE).astype(np.float32)
    embedder.settings.return_value = {"embedder": "mock"}
    embedder.output_shape = TEST_EMBEDDING_SIZE
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
def pipeline_full(mock_reporter, mock_detector, mock_embedder, mock_classifier):
    """Create a full pipeline with all mocked components."""
    # Mock the preprocessor for crop functionality
    with patch('lib.API.Preprocessor.Preprocessor.crop') as mock_crop:
        mock_crop.return_value = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        pipeline = Pipeline(
            reporter=mock_reporter,
            detector=mock_detector,
            embedder=mock_embedder,
            classifier=mock_classifier
        )
        # Add preprocessor to pipeline
        pipeline.preprocessor = Mock()
        pipeline.preprocessor.load.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pipeline.preprocessor.crop.return_value = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        return pipeline


@pytest.fixture
def pipeline_full_real(reporter):
    """Create a pipeline with real components (if available)."""
    try:
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
    except ImportError:
        # Return mock pipeline if real components not available
        return pipeline_full(reporter, Mock(), Mock(), Mock())


@pytest.fixture
def dummy_image_path(tmp_path):
    """Create a dummy image file."""
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), dummy_image)
    return image_path


@pytest.fixture
def multiple_dummy_images(tmp_path):
    """Create multiple dummy image files."""
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    images = []
    for i in range(3):
        image_path = tmp_path / f"test_image_{i}.jpg"
        cv2.imwrite(str(image_path), dummy_image)
        images.append(image_path)
    return images