"""
Comprehensive test suite for the Reporter class and ReporterConfig.
Tests all core functionality including configuration management, directory operations,
file saving in various formats, visualization features, and integration workflows.
"""

import numpy as np
import pandas as pd
import json
import pytest
import cv2
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lib.API.Reporter import Reporter, ReporterConfig, OutputFormat, ModelFormat
from lib.face_classification.KMeansClassifier import KMeansClassifier


@pytest.mark.reporter
@pytest.mark.config
class TestReporterConfig:
    """Test ReporterConfig dataclass configuration and validation."""
    
    def test_default_config_creation(self, tmp_path):
        """Test creating ReporterConfig with default values."""
        config = ReporterConfig(output_dir=tmp_path)
        
        assert config.output_dir == tmp_path
        assert config.save_annotated_image is True
        assert config.save_cropped_faces is True
        assert config.save_model is True
        assert config.save_model_settings is True
        assert config.save_image_results_to_file is True
        assert config.save_compiled_results is True
        assert config.save_cosine_similarity_matrix is True
        assert config.save_cosine_similarity_matrix_visualization is True
        assert config.save_tsne_visualization is True
        
    def test_custom_config_creation(self, tmp_path):
        """Test creating ReporterConfig with custom values."""
        config = ReporterConfig(
            output_dir=tmp_path,
            save_annotated_image=False,
            save_cropped_faces=False,
            save_model=False,
            save_model_settings_format=OutputFormat.JSON,
            save_image_results_to_file_format=OutputFormat.TXT
        )
        
        assert config.save_annotated_image is False
        assert config.save_cropped_faces is False
        assert config.save_model is False
        assert config.save_model_settings_format == OutputFormat.JSON
        assert config.save_image_results_to_file_format == OutputFormat.TXT
        
    def test_is_saving_enabled_property(self, tmp_path):
        """Test the is_saving_enabled property logic."""
        # All disabled
        config = ReporterConfig(
            output_dir=tmp_path,
            save_annotated_image=False,
            save_cropped_faces=False,
            save_model=False,
            save_model_settings=False,
            save_image_results_to_file=False,
            save_compiled_results=False
        )
        assert config.is_saving_enabled is False
        
        # At least one enabled
        config.save_annotated_image = True
        assert config.is_saving_enabled is True


@pytest.mark.reporter
@pytest.mark.unit
class TestReporterInitialization:
    """Test Reporter class initialization and basic setup."""
    
    def test_reporter_initialization(self, tmp_path):
        """Test Reporter initialization with config."""
        config = ReporterConfig(output_dir=tmp_path)
        reporter = Reporter(config)
        
        assert reporter.config == config
        assert reporter.output_dir == tmp_path
        assert reporter.bulk_mode is False
        
    def test_reporter_initialization_custom_config(self, tmp_path):
        """Test Reporter initialization with custom config."""
        config = ReporterConfig(
            output_dir=tmp_path,
            save_annotated_image=False,
            save_model_format=ModelFormat.ONNX
        )
        reporter = Reporter(config)
        
        assert reporter.config.save_annotated_image is False
        assert reporter.config.save_model_format == ModelFormat.ONNX


@pytest.mark.reporter
@pytest.mark.unit
class TestReporterDirectoryManagement:
    """Test Reporter directory creation and management methods."""
    
    def test_setup_output_directory_single_mode(self, reporter, tmp_path):
        """Test output directory setup for single image processing."""
        img_path = tmp_path / "test_image.jpg"
        reporter.setup_output_directory(img_path)
        
        assert reporter.output_dir_result.exists()
        assert "test_image" in str(reporter.output_dir_result)
        
    def test_setup_bulk_output_directory(self, reporter):
        """Test bulk output directory setup."""
        reporter.setup_bulk_output_directory()
        
        assert reporter.bulk_mode is True
        assert reporter.output_dir.exists()
        assert "Bulk_" in str(reporter.output_dir)
        
    def test_setup_training_output_directory(self, reporter):
        """Test training output directory setup."""
        reporter.setup_training_output_directory()
        
        assert reporter.output_dir.exists()
        assert "Training_" in str(reporter.output_dir)
        
    def test_bulk_mode_output_structure(self, reporter, tmp_path):
        """Test that bulk mode creates correct subdirectory structure."""
        reporter.setup_bulk_output_directory()
        
        img_path = tmp_path / "bulk_test.jpg"
        reporter.setup_output_directory(img_path)
        
        # In bulk mode, should create subdirectory with image name
        expected_subdir = reporter.output_dir / "bulk_test"
        assert reporter.output_dir_result == expected_subdir
        assert expected_subdir.exists()


@pytest.mark.reporter
@pytest.mark.unit
class TestReporterDataValidation:
    """Test Reporter input validation and edge cases."""
    
    def test_save_with_empty_results(self, reporter, dummy_image, tmp_path):
        """Test saving with empty results list."""
        class MockModel:
            def settings(self): return {"test": "value"}
            
        img_path = tmp_path / "empty_test.jpg"
        cv2.imwrite(str(img_path), dummy_image)
        
        # Should not crash with empty results
        reporter.save_processed_image_results(
            MockModel(), MockModel(), MockModel(),
            dummy_image.copy(), img_path, results=[]
        )
        
    def test_save_with_none_cropped_faces(self, reporter, dummy_image, dummy_results, tmp_path):
        """Test saving with None cropped faces."""
        class MockModel:
            def settings(self): return {"test": "value"}
            
        img_path = tmp_path / "none_faces.jpg"
        cv2.imwrite(str(img_path), dummy_image)
        
        # Should not crash with None cropped faces
        reporter.save_processed_image_results(
            MockModel(), MockModel(), MockModel(),
            dummy_image.copy(), img_path, 
            results=dummy_results.copy(), cropped_faces=None
        )
        
    def test_save_with_disabled_saving(self, tmp_path, dummy_image, dummy_results):
        """Test that saving is skipped when disabled in config."""
        config = ReporterConfig(
            output_dir=tmp_path,
            save_annotated_image=False,
            save_cropped_faces=False,
            save_model=False,
            save_model_settings=False,
            save_image_results_to_file=False,
            save_compiled_results=False
        )
        reporter = Reporter(config)
        
        class MockModel:
            def settings(self): return {"test": "value"}
            
        img_path = tmp_path / "disabled_test.jpg"
        cv2.imwrite(str(img_path), dummy_image)
        
        # Should exit early due to is_saving_enabled being False
        reporter.save_processed_image_results(
            MockModel(), MockModel(), MockModel(),
            dummy_image.copy(), img_path, results=dummy_results.copy()
        )
        
        # No files should be created
        assert not list(tmp_path.rglob("*_results.*"))


@pytest.mark.reporter
@pytest.mark.unit
class TestReporterFileSaving:
    """Test Reporter file saving methods in different formats."""
    
    def test_save_to_file_json_format(self, reporter, dummy_results, tmp_path):
        """Test saving results to JSON format."""
        reporter.output_dir_result = tmp_path
        
        reporter.save_to_file(
            tmp_path / "test_results.json", 
            dummy_results, 
            format=OutputFormat.JSON
        )
        
        saved_file = tmp_path / "test_results.json"
        assert saved_file.exists()
        
        with open(saved_file, 'r') as f:
            loaded_data = json.load(f)
        assert len(loaded_data) == len(dummy_results)
        
    def test_save_to_file_csv_format(self, reporter, dummy_results, tmp_path):
        """Test saving results to CSV format."""
        reporter.output_dir_result = tmp_path
        
        reporter.save_to_file(
            tmp_path / "test_results.csv", 
            dummy_results, 
            format=OutputFormat.CSV
        )
        
        saved_file = tmp_path / "test_results.csv"
        assert saved_file.exists()
        
        df = pd.read_csv(saved_file)
        assert len(df) == len(dummy_results)
        
    def test_save_to_file_txt_format(self, reporter, dummy_results, tmp_path):
        """Test saving results to TXT format."""
        reporter.output_dir_result = tmp_path
        
        reporter.save_to_file(
            tmp_path / "test_results.txt", 
            dummy_results, 
            format=OutputFormat.TXT
        )
        
        saved_file = tmp_path / "test_results.txt"
        assert saved_file.exists()
        
        with open(saved_file, 'r') as f:
            content = f.read()
        assert "Result 0:" in content
        assert "bbox" in content
        
    def test_save_to_file_bin_format(self, reporter, dummy_results, tmp_path):
        """Test saving results to binary format."""
        reporter.output_dir_result = tmp_path
        
        reporter.save_to_file(
            tmp_path / "test_results.npz", 
            dummy_results, 
            format=OutputFormat.BIN
        )
        
        saved_file = tmp_path / "test_results.npz"
        assert saved_file.exists()
        
        loaded_data = np.load(saved_file, allow_pickle=True)
        assert 'bbox' in loaded_data
        
    def test_save_to_file_parquet_format(self, reporter, dummy_results, tmp_path):
        """Test saving results to Parquet format."""
        reporter.output_dir_result = tmp_path
        
        reporter.save_to_file(
            tmp_path / "test_results.parquet", 
            dummy_results, 
            format=OutputFormat.PARQUET
        )
        
        saved_file = tmp_path / "test_results.parquet"
        assert saved_file.exists()
        
        df = pd.read_parquet(saved_file)
        assert len(df) == len(dummy_results)
        
    def test_save_to_file_unsupported_format(self, reporter, dummy_results, tmp_path):
        """Test error handling for unsupported format."""
        reporter.output_dir_result = tmp_path
        
        # Create a mock unsupported format
        with pytest.raises(ValueError, match="Unsupported output format"):
            reporter.save_to_file(
                tmp_path / "test.unknown", 
                dummy_results, 
                format="UNKNOWN"
            )
    
    def test_save_face_images(self, reporter, dummy_image, dummy_results, tmp_path):
        """Test saving cropped face images."""
        reporter.output_dir_result = tmp_path
        img_path = tmp_path / "test_face.jpg"
        
        faces = [dummy_image.copy(), dummy_image.copy()]
        reporter.save_face_images(img_path, "cropped", faces, dummy_results.copy())
        
        cropped_dir = tmp_path / "cropped"
        assert cropped_dir.exists()
        
        face_files = list(cropped_dir.glob("*.jpg"))
        assert len(face_files) == 2
        
        # Check that face_image references were added to results
        for i, result in enumerate(dummy_results):
            if i < len(faces):
                assert "face_image" in result
    
    def test_save_model_settings(self, reporter, tmp_path):
        """Test saving model settings."""
        reporter.output_dir_result = tmp_path
        
        class MockDetector:
            def settings(self): return {"detector_param": "value1"}
            
        class MockEmbedder:
            def settings(self): return {"embedder_param": "value2"}
            
        class MockClassifier:
            def settings(self): return {"classifier_param": "value3"}
        
        reporter.save_model_settings(MockDetector(), MockEmbedder(), MockClassifier())
        
        settings_file = tmp_path / "model_settings.json"
        assert settings_file.exists()
        
        with open(settings_file, 'r') as f:
            settings = json.load(f)
        
        assert len(settings) == 3
        assert settings[0]["detector"]["detector_param"] == "value1"
        assert settings[1]["embedder"]["embedder_param"] == "value2"
        assert settings[2]["classifier"]["classifier_param"] == "value3"


@pytest.mark.reporter
@pytest.mark.visualization
@pytest.mark.slow
class TestReporterVisualizationMethods:
    """Test Reporter visualization and analysis methods."""
    
    def test_save_cosine_similarity_matrix(self, reporter, dummy_results, tmp_path):
        """Test cosine similarity matrix computation and saving."""
        reporter.output_dir_result = tmp_path
        img_path = tmp_path / "test_similarity.jpg"
        
        # Ensure we have enough embeddings for similarity computation
        results_with_embeddings = []
        for i in range(3):
            result = dummy_results[i % len(dummy_results)].copy()
            result["embedding"] = np.random.rand(128).astype(np.float32)
            results_with_embeddings.append(result)
        
        reporter.save_cosine_similarity_matrix(img_path, results_with_embeddings)
        
        similarity_file = tmp_path / "cosine_similarity_matrix_test_similarity.csv"
        assert similarity_file.exists()
        
        df = pd.read_csv(similarity_file)
        assert df.shape[0] == df.shape[1] == 3  # Square matrix
        
    def test_save_cosine_similarity_matrix_insufficient_embeddings(self, reporter, tmp_path):
        """Test cosine similarity with insufficient embeddings."""
        reporter.output_dir_result = tmp_path
        img_path = tmp_path / "insufficient.jpg"
        
        # Only one embedding - should log warning and return
        single_result = [{
            "bbox": (10, 20, 50, 60),
            "embedding": np.random.rand(128).astype(np.float32)
        }]
        
        # Should not crash, but no file should be created
        reporter.save_cosine_similarity_matrix(img_path, single_result)
        
        similarity_files = list(tmp_path.glob("*similarity*.csv"))
        assert len(similarity_files) == 0
        
    def test_save_tsne_visualization(self, reporter, tmp_path):
        """Test t-SNE visualization saving."""
        # Create test data with embeddings
        test_data = []
        for i in range(10):  # Need enough points for t-SNE
            test_data.append({
                "embedding": np.random.rand(128).astype(np.float32),
                "label": f"Person{i % 3}"
            })
        
        path = tmp_path / "tsne_test"
        reporter.output_dir = tmp_path
        
        reporter.save_tsne_visualization(path, test_data)
        
        tsne_file = tmp_path / "tsne_visualization_tsne_test.svg"
        assert tsne_file.exists()
        
    def test_save_tsne_insufficient_data(self, reporter, tmp_path):
        """Test t-SNE with insufficient data points."""
        test_data = [{"embedding": np.random.rand(128).astype(np.float32)}]
        
        path = tmp_path / "tsne_insufficient"
        reporter.output_dir = tmp_path
        
        # Should handle gracefully with insufficient data
        reporter.save_tsne_visualization(path, test_data)


@pytest.mark.reporter
@pytest.mark.integration
class TestReporterBulkOperations:
    """Test Reporter bulk processing and compilation methods."""
    
    def test_compile_all_results_in_bulk_mode(self, reporter, dummy_results, tmp_path):
        """Test compiling all results in bulk mode."""
        reporter.setup_bulk_output_directory()
        
        # Create test subdirectories with results
        for i in range(3):
            subdir = reporter.output_dir / f"img{i}"
            subdir.mkdir()
            
            # Save results with embeddings
            results_with_embeddings = []
            for result in dummy_results:
                result_copy = result.copy()
                # Create embedding file reference
                emb_dir = subdir / "embeddings"
                emb_dir.mkdir(exist_ok=True)
                emb_file = emb_dir / f"embedding_img{i}_0.npy"
                np.save(emb_file, result["embedding"])
                result_copy["embedding_file"] = str(emb_file.relative_to(reporter.output_dir))
                # Remove the embedding array to simulate the real workflow
                del result_copy["embedding"]
                results_with_embeddings.append(result_copy)
            
            results_file = subdir / f"img{i}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results_with_embeddings, f)
        
        reporter.compile_all_results()
        
        compiled_file = reporter.output_dir / "compiled_results.parquet"
        assert compiled_file.exists()
        
        df = pd.read_parquet(compiled_file)
        assert len(df) > 0
        
    def test_compile_all_results_not_in_bulk_mode(self, reporter):
        """Test that compilation is skipped when not in bulk mode."""
        # Should log info and return early
        reporter.compile_all_results()
        # No error should be raised
        
    def test_save_batch_summary(self, reporter, tmp_path):
        """Test saving batch processing summary."""
        reporter.output_dir = tmp_path
        
        batch_result = {
            "total_images": 10,
            "total_faces": 25,
            "processing_time": 45.2,
            "average_faces_per_image": 2.5
        }
        
        reporter.save_batch_summary(batch_result)
        
        summary_file = tmp_path / "batch_summary.json"
        assert summary_file.exists()
        
        with open(summary_file, 'r') as f:
            saved_summary = json.load(f)
        
        assert saved_summary[0]["total_images"] == 10
        assert saved_summary[0]["total_faces"] == 25


@pytest.mark.reporter
@pytest.mark.integration
class TestReporterTrainingWorkflow:
    """Test Reporter training-related functionality."""
    
    def test_save_test_summary(self, reporter, tmp_path):
        """Test saving test summary for training."""
        reporter.output_dir = tmp_path
        
        test_result = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.92,
            "f1_score": 0.925
        }
        
        train_data = np.random.rand(10, 128).astype(np.float32)
        
        reporter.save_test_summary(
            test_result=test_result,
            classifier=None,
            train_data=train_data
        )
        
        summary_file = reporter.output_dir / "test_summary.json"
        assert summary_file.exists()
        
        with open(summary_file, 'r') as f:
            saved_result = json.load(f)
        
        assert saved_result[0]["accuracy"] == 0.95
        
    def test_save_test_summary_with_classifier(self, tmp_path):
        """Test saving test summary with KMeans classifier."""
        config = ReporterConfig(output_dir=tmp_path, save_model=True)
        reporter = Reporter(config)
        
        # Create a mock classifier
        mock_classifier = Mock()
        mock_classifier.cluster_centers = np.random.rand(3, 128).astype(np.float32)
        mock_classifier.get_name.return_value = "MockKMeans"
        mock_classifier.model = Mock()
        
        test_result = {"accuracy": 0.9}
        train_data = np.random.rand(10, 128).astype(np.float32)
        
        with patch('lib.API.Reporter.to_onnx') as mock_to_onnx:
            mock_to_onnx.return_value.SerializeToString.return_value = b"mock_onnx_data"
            
            reporter.save_test_summary(
                test_result=test_result,
                classifier=mock_classifier,
                train_data=train_data
            )
        
        # Check that cluster centers were saved
        assert (reporter.output_dir / "cluster_centers.npy").exists()
        assert (reporter.output_dir / "cluster_centers.parquet").exists()


@pytest.mark.reporter
@pytest.mark.unit
class TestReporterAnnotationMethods:
    """Test Reporter image annotation and processing methods."""
    
    def test_save_annotated_image(self, reporter, dummy_image, dummy_results, tmp_path):
        """Test saving annotated image with bounding boxes."""
        reporter.output_dir_result = tmp_path
        img_path = tmp_path / "test_annotation.jpg"
        
        class MockModel:
            def __init__(self, name):
                self.name = name
            def __class__(self):
                return type(self.name, (), {})()
            @property
            def __class__(self):
                return type(self.name, (), {})
        
        # Create mock models
        detector = MockModel("MockDetector")
        embedder = MockModel("MockEmbedder")  
        classifier = MockModel("MockClassifier")
        
        # Test image should be modified with annotations
        test_image = dummy_image.copy()
        
        reporter.save_annotated_image(
            detector, embedder, classifier,
            img_path, test_image, dummy_results.copy()
        )
        
        annotated_file = tmp_path / "test_annotation_annotated.jpg"
        assert annotated_file.exists()
        
        # Check that the image file has content
        assert annotated_file.stat().st_size > 0
        
    def test_save_annotated_image_with_landmarks(self, reporter, dummy_image, tmp_path):
        """Test annotated image saving with landmark data."""
        reporter.output_dir_result = tmp_path
        img_path = tmp_path / "test_landmarks.jpg"
        
        # Create results with landmarks
        results_with_landmarks = [{
            "bbox": (10, 20, 50, 60),
            "score": 0.95,
            "landmarks": [[25, 30], [35, 30], [30, 40], [25, 45], [35, 45]],  # 5 face landmarks
            "label": "TestPerson"
        }]
        
        class MockModel:
            def __init__(self, name):
                self.name = name
            @property 
            def __class__(self):
                return type(self.name, (), {})
        
        detector = MockModel("MockDetector")
        embedder = MockModel("MockEmbedder")
        classifier = MockModel("MockClassifier")
        
        test_image = dummy_image.copy()
        
        reporter.save_annotated_image(
            detector, embedder, classifier,
            img_path, test_image, results_with_landmarks
        )
        
        annotated_file = tmp_path / "test_landmarks_annotated.jpg"
        assert annotated_file.exists()


@pytest.mark.reporter
@pytest.mark.integration
class TestReporterIntegration:
    """Integration tests for complete Reporter workflows."""
    
    def test_complete_single_image_workflow(self, reporter, dummy_image, dummy_results, tmp_path):
        """Test complete workflow for single image processing."""
        class MockModel:
            def settings(self): return {"param": "test_value"}
            def get_name(self): return "MockModel"
        
        img_path = tmp_path / "integration_test.jpg"
        cv2.imwrite(str(img_path), dummy_image)
        
        detector = embedder = classifier = MockModel()
        cropped_faces = [dummy_image.copy() for _ in range(len(dummy_results))]
        
        # Run complete workflow
        reporter.save_processed_image_results(
            detector, embedder, classifier,
            dummy_image.copy(), img_path,
            results=dummy_results.copy(),
            cropped_faces=cropped_faces
        )
        
        # Verify all expected outputs
        outdir = reporter.output_dir_result
        assert outdir.exists()
        assert (outdir / "model_settings.json").exists()
        assert (outdir / "integration_test_annotated.jpg").exists()
        assert (outdir / "cropped").exists()
        assert (outdir / "embeddings").exists()
        
        # Check that results file was created
        results_files = list(outdir.glob("*_results.*"))
        assert len(results_files) > 0
        
        # Check embeddings were saved separately
        embedding_files = list((outdir / "embeddings").glob("*.npy"))
        assert len(embedding_files) == len(dummy_results)
        
    def test_complete_bulk_workflow(self, reporter, dummy_image, dummy_results, tmp_path):
        """Test complete workflow for bulk processing."""
        class MockModel:
            def settings(self): return {"param": "bulk_test"}
            def get_name(self): return "BulkMockModel"
        
        reporter.setup_bulk_output_directory()
        
        # Process multiple images
        for i in range(3):
            img_path = tmp_path / f"bulk_img_{i}.jpg"
            cv2.imwrite(str(img_path), dummy_image)
            
            reporter.save_processed_image_results(
                MockModel(), MockModel(), MockModel(),
                dummy_image.copy(), img_path,
                results=dummy_results.copy(),
                cropped_faces=[dummy_image.copy() for _ in range(len(dummy_results))]
            )
        
        # Compile results
        reporter.compile_all_results()
        
        # Verify bulk structure
        assert reporter.bulk_mode is True
        assert (reporter.output_dir / "compiled_results.parquet").exists()
        
        # Check individual image subdirectories
        for i in range(3):
            img_subdir = reporter.output_dir / f"bulk_img_{i}"
            assert img_subdir.exists()
            assert list(img_subdir.glob("*_results.*"))


@pytest.mark.reporter
@pytest.mark.unit
class TestReporterErrorHandling:
    """Test Reporter error handling and edge cases."""
    
    def test_save_without_output_directory_setup(self, tmp_path):
        """Test error when trying to save without setting up output directory."""
        config = ReporterConfig(output_dir=tmp_path)
        reporter = Reporter(config)
        
        # Trying to save without setup should raise RuntimeError
        with pytest.raises(RuntimeError, match="Output directory structure not created"):
            reporter.save_to_file(Path("test.json"), [{"test": "data"}])
            
    def test_save_empty_data_warning(self, reporter, tmp_path):
        """Test warning when saving empty data."""
        reporter.output_dir_result = tmp_path
        
        # Should log warning and skip saving
        reporter.save_to_file(tmp_path / "empty.json", [])
        
        # File should not be created
        assert not (tmp_path / "empty.json").exists()
        
    def test_malformed_results_handling(self, reporter, tmp_path):
        """Test handling of malformed results data."""
        reporter.output_dir_result = tmp_path
        
        # Results missing expected keys
        malformed_results = [
            {"bbox": (10, 20, 30, 40)},  # Missing other expected keys
            {"score": 0.5},              # Missing bbox
            {}                           # Empty result
        ]
        
        # Should not crash when saving malformed data
        reporter.save_to_file(
            tmp_path / "malformed.json", 
            malformed_results, 
            format=OutputFormat.JSON
        )
        
        assert (tmp_path / "malformed.json").exists()


# Legacy tests for backward compatibility
def test_save_processed_image_results(reporter, dummy_image, dummy_results, tmp_path):
    """Legacy test for backward compatibility."""
    class Dummy:
        def settings(self): return {"param": "value"}
    detector = embedder = classifier = Dummy()

    img_path = tmp_path / "image1.jpg"
    cv2.imwrite(str(img_path), dummy_image)

    reporter.save_processed_image_results(
        detector, embedder, classifier,
        dummy_image.copy(), img_path,
        results=dummy_results.copy(),
        cropped_faces=[dummy_image.copy(), dummy_image.copy()]
    )

    outdir = reporter.output_dir_result
    assert (outdir / "model_settings.json").exists()
    assert any(f.name.endswith("_results.csv") for f in outdir.iterdir())
    assert (outdir / "image1_annotated.jpg").exists()
    assert (outdir / "cropped").exists()
    assert (outdir / "embeddings").exists()


def test_bulk_mode(reporter, dummy_image, dummy_results, tmp_path):
    """Legacy test for backward compatibility."""
    reporter.setup_bulk_output_directory()
    img_path = tmp_path / "bulk_img.jpg"
    cv2.imwrite(str(img_path), dummy_image)

    class Dummy:
        def settings(self): return {"param": "value"}
    detector = embedder = classifier = Dummy()

    reporter.save_processed_image_results(
        detector, embedder, classifier,
        dummy_image.copy(), img_path,
        results=dummy_results.copy(),
        cropped_faces=[dummy_image.copy(), dummy_image.copy()]
    )

    img_subdir = reporter.output_dir / "bulk_img"
    assert img_subdir.exists()
    assert any(f.suffix in [".json", ".csv"] for f in img_subdir.iterdir())


def test_compile_results(reporter, dummy_results, tmp_path):
    """Legacy test for backward compatibility."""
    reporter.setup_bulk_output_directory()
    subdir = reporter.output_dir / "img1"
    subdir.mkdir()
    results_file = subdir / "img1_results.json"

    with open(results_file, "w") as f:
        json.dump(dummy_results, f)

    reporter.compile_all_results()
    compiled_file = reporter.output_dir / "compiled_results.parquet"
    assert compiled_file.exists()


def test_training_output(reporter, tmp_path):
    """Legacy test for backward compatibility."""
    reporter.setup_training_output_directory()
    assert reporter.output_dir.exists()
    reporter.save_test_summary(
        test_result={"accuracy": 0.9},
        classifier=None,
        train_data=np.random.rand(5, 128).astype(np.float32),
    )
    assert (reporter.output_dir / "test_summary.json").exists()


def test_no_results_does_not_crash(reporter, dummy_image, tmp_path):
    """Legacy test for backward compatibility."""
    class Dummy:
        def settings(self): return {}
    img_path = tmp_path / "img.jpg"
    cv2.imwrite(str(img_path), dummy_image)
    reporter.save_processed_image_results(
        Dummy(), Dummy(), Dummy(), dummy_image.copy(), img_path, results=[]
    )