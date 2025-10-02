"""
Simplified test suite for the Reporter class focusing on core saving capabilities.
Tests all output structures, configuration states, and input types for the main
user-facing functionality: saving results in all formats and configurations.
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


@pytest.mark.reporter
@pytest.mark.unit
class TestReporterConfiguration:
    """Test ReporterConfig and basic Reporter setup scenarios."""
    
    def test_default_reporter_configuration(self, tmp_path):
        """Test Reporter with default configuration settings."""
        config = ReporterConfig(output_dir=tmp_path)
        reporter = Reporter(config)
        
        # Verify default settings
        assert config.save_annotated_image is True
        assert config.save_cropped_faces is True
        assert config.save_model is True
        assert config.save_model_settings is True
        assert config.save_image_results_to_file is True
        assert config.save_compiled_results is True
        assert config.is_saving_enabled is True
        
        # Verify reporter initialization
        assert reporter.config == config
        assert reporter.output_dir == tmp_path
        assert reporter.bulk_mode is False
    
    def test_custom_reporter_configuration(self, tmp_path):
        """Test Reporter with custom configuration settings."""
        config = ReporterConfig(
            output_dir=tmp_path,
            save_annotated_image=False,
            save_cropped_faces=False,
            save_model_format=ModelFormat.ONNX,
            save_image_results_to_file_format=OutputFormat.JSON
        )
        reporter = Reporter(config)
        
        assert config.save_annotated_image is False
        assert config.save_cropped_faces is False
        assert config.save_model_format == ModelFormat.ONNX
        assert config.save_image_results_to_file_format == OutputFormat.JSON
    
    def test_saving_disabled_configuration(self, tmp_path):
        """Test Reporter with all saving disabled."""
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


@pytest.mark.reporter
@pytest.mark.integration
class TestReporterSingleImageWorkflow:
    """Test complete single image processing workflow with various configurations."""
    
    def test_single_image_all_features_enabled(self, reporter, dummy_image, dummy_results, tmp_path):
        """Test single image processing with all features enabled."""
        # Create mock models
        class MockModel:
            def settings(self): return {"param": "test_value"}
        
        detector = embedder = classifier = MockModel()
        
        # Create test image file
        img_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(img_path), dummy_image)
        
        # Create cropped faces
        cropped_faces = [dummy_image.copy() for _ in range(len(dummy_results))]
        
        # Process image
        reporter.save_processed_image_results(
            detector, embedder, classifier,
            dummy_image.copy(), img_path,
            results=dummy_results.copy(),
            cropped_faces=cropped_faces
        )
        
        # Verify output structure
        output_dir = reporter.output_dir_result
        assert output_dir.exists()
        
        # Check all expected files and directories
        assert (output_dir / "model_settings.json").exists()
        assert (output_dir / "test_image_annotated.jpg").exists()
        assert (output_dir / "cropped").exists()
        assert (output_dir / "embeddings").exists()
        
        # Check results file was created
        results_files = list(output_dir.glob("*_results.*"))
        assert len(results_files) > 0
        
        # Check embedding files were created
        embedding_files = list((output_dir / "embeddings").glob("*.npy"))
        assert len(embedding_files) == len(dummy_results)
        
        # Check cropped face files
        cropped_files = list((output_dir / "cropped").glob("*.jpg"))
        assert len(cropped_files) == len(cropped_faces)
    
    def test_single_image_minimal_configuration(self, tmp_path, dummy_image, dummy_results):
        """Test single image processing with minimal saving configuration."""
        config = ReporterConfig(
            output_dir=tmp_path,
            save_annotated_image=False,
            save_cropped_faces=False,
            save_model_settings=True,
            save_image_results_to_file=True
        )
        reporter = Reporter(config)
        
        class MockModel:
            def settings(self): return {"minimal": "config"}
        
        detector = embedder = classifier = MockModel()
        img_path = tmp_path / "minimal_test.jpg"
        cv2.imwrite(str(img_path), dummy_image)
        
        reporter.save_processed_image_results(
            detector, embedder, classifier,
            dummy_image.copy(), img_path,
            results=dummy_results.copy()
        )
        
        # Should only have model settings and results file
        output_dir = reporter.output_dir_result
        assert (output_dir / "model_settings.json").exists()
        assert list(output_dir.glob("*_results.*"))
        
        # Should not have annotated image or cropped faces
        assert not (output_dir / "minimal_test_annotated.jpg").exists()
        assert not (output_dir / "cropped").exists()
    
    def test_single_image_no_results(self, reporter, dummy_image, tmp_path):
        """Test single image processing with no detection results."""
        class MockModel:
            def settings(self): return {"empty": "results"}
        
        detector = embedder = classifier = MockModel()
        img_path = tmp_path / "no_faces.jpg"
        cv2.imwrite(str(img_path), dummy_image)
        
        # Process with empty results
        reporter.save_processed_image_results(
            detector, embedder, classifier,
            dummy_image.copy(), img_path,
            results=[]  # No faces detected
        )
        
        # Should still create output directory and model settings
        output_dir = reporter.output_dir_result
        assert output_dir.exists()
        assert (output_dir / "model_settings.json").exists()


@pytest.mark.reporter
@pytest.mark.integration
class TestReporterBulkWorkflow:
    """Test bulk processing workflow and results compilation."""
    
    def test_bulk_processing_complete_workflow(self, reporter, dummy_image, dummy_results, tmp_path):
        """Test complete bulk processing workflow."""
        # Set up bulk mode
        reporter.setup_bulk_output_directory()
        assert reporter.bulk_mode is True
        assert "Bulk_" in str(reporter.output_dir)
        
        class MockModel:
            def settings(self): return {"bulk": "processing"}
        
        detector = embedder = classifier = MockModel()
        
        # Process multiple images
        for i in range(3):
            img_path = tmp_path / f"bulk_image_{i}.jpg"
            cv2.imwrite(str(img_path), dummy_image)
            
            reporter.save_processed_image_results(
                detector, embedder, classifier,
                dummy_image.copy(), img_path,
                results=dummy_results.copy(),
                cropped_faces=[dummy_image.copy() for _ in range(len(dummy_results))]
            )
        
        # Verify bulk structure
        for i in range(3):
            img_subdir = reporter.output_dir / f"bulk_image_{i}"
            assert img_subdir.exists()
            assert list(img_subdir.glob("*_results.*"))
        
        # Test results compilation
        reporter.compile_all_results()
        compiled_file = reporter.output_dir / "compiled_results.parquet"
        assert compiled_file.exists()
        
        # Verify compiled data
        df = pd.read_parquet(compiled_file)
        assert len(df) > 0
        
        # Test batch summary
        batch_summary = {
            "total_images": 3,
            "total_faces": 12,
            "processing_time": 45.2
        }
        reporter.save_batch_summary(batch_summary)
        
        summary_file = reporter.output_dir / "batch_summary.json"
        assert summary_file.exists()
        
        with open(summary_file, 'r') as f:
            saved_summary = json.load(f)
        assert saved_summary[0]["total_images"] == 3
    
    def test_bulk_not_in_bulk_mode(self, reporter):
        """Test that compilation is skipped when not in bulk mode."""
        # Should handle gracefully when not in bulk mode
        reporter.compile_all_results()
        # No error should be raised


@pytest.mark.reporter
@pytest.mark.integration
class TestReporterTrainingWorkflow:
    """Test training workflow and model saving."""
    
    def test_training_workflow_with_model_saving(self, tmp_path):
        """Test complete training workflow with model saving."""
        config = ReporterConfig(output_dir=tmp_path, save_model=True)
        reporter = Reporter(config)
        
        # Create mock classifier with cluster centers
        mock_classifier = Mock()
        mock_classifier.cluster_centers = np.random.rand(3, 128).astype(np.float32)
        mock_classifier.get_name.return_value = "TestKMeans"
        mock_classifier.model = Mock()
        
        test_result = {
            "accuracy": 0.95,
            "n_clusters": 3,
            "silhouette_score": 0.7
        }
        train_data = np.random.rand(10, 128).astype(np.float32)
        
        # Mock ONNX conversion
        with patch('lib.API.Reporter.to_onnx') as mock_to_onnx:
            mock_to_onnx.return_value.SerializeToString.return_value = b"mock_onnx_data"
            
            reporter.save_test_summary(
                test_result=test_result,
                classifier=mock_classifier,
                train_data=train_data,
                silhouette_scores=[0.6, 0.7, 0.65],
                inertias=[100, 80, 90]
            )
        
        # Verify training output structure
        assert "Training_" in str(reporter.output_dir)
        assert (reporter.output_dir / "test_summary.json").exists()
        assert (reporter.output_dir / "cluster_centers.npy").exists()
        assert (reporter.output_dir / "cluster_centers.parquet").exists()
        
        # Verify test summary content
        with open(reporter.output_dir / "test_summary.json", 'r') as f:
            saved_result = json.load(f)
        assert saved_result[0]["accuracy"] == 0.95
    
    def test_training_workflow_without_model_saving(self, tmp_path):
        """Test training workflow with model saving disabled."""
        config = ReporterConfig(output_dir=tmp_path, save_model=False)
        reporter = Reporter(config)
        
        test_result = {"accuracy": 0.8}
        train_data = np.random.rand(5, 128).astype(np.float32)
        
        reporter.save_test_summary(
            test_result=test_result,
            classifier=None,
            train_data=train_data
        )
        
        # Should only create test summary
        assert (reporter.output_dir / "test_summary.json").exists()
        assert not (reporter.output_dir / "cluster_centers.npy").exists()


@pytest.mark.reporter
@pytest.mark.unit
class TestReporterFileFormats:
    """Test all supported file formats for saving results."""
    
    def test_all_output_formats(self, reporter, dummy_results, tmp_path):
        """Test saving results in all supported output formats."""
        reporter.output_dir_result = tmp_path
        
        formats_to_test = [
            (OutputFormat.JSON, "results.json"),
            (OutputFormat.CSV, "results.csv"),
            (OutputFormat.TXT, "results.txt"),
            (OutputFormat.BIN, "results.npz"),
            (OutputFormat.PARQUET, "results.parquet")
        ]
        
        for format_type, filename in formats_to_test:
            file_path = tmp_path / filename
            
            # Test saving
            reporter.save_to_file(file_path, dummy_results, format_type)
            
            # Verify file was created
            assert file_path.exists()
            assert file_path.stat().st_size > 0
            
            # Basic format-specific validation
            if format_type == OutputFormat.JSON:
                with open(file_path, 'r') as f:
                    loaded_data = json.load(f)
                assert len(loaded_data) == len(dummy_results)
            
            elif format_type == OutputFormat.CSV:
                df = pd.read_csv(file_path)
                assert len(df) == len(dummy_results)
            
            elif format_type == OutputFormat.PARQUET:
                df = pd.read_parquet(file_path)
                assert len(df) == len(dummy_results)
            
            elif format_type == OutputFormat.BIN:
                loaded_data = np.load(file_path, allow_pickle=True)
                assert 'bbox' in loaded_data
    
    def test_unsupported_format_error(self, reporter, dummy_results, tmp_path):
        """Test error handling for unsupported file format."""
        reporter.output_dir_result = tmp_path
        
        with pytest.raises(ValueError, match="Unsupported output format"):
            reporter.save_to_file(tmp_path / "test.unknown", dummy_results, "UNKNOWN_FORMAT")
    
    def test_empty_data_handling(self, reporter, tmp_path):
        """Test handling of empty data during saving."""
        reporter.output_dir_result = tmp_path
        
        # Should handle empty data gracefully
        reporter.save_to_file(tmp_path / "empty.json", [], OutputFormat.JSON)
        
        # File should not be created for empty data
        assert not (tmp_path / "empty.json").exists()


@pytest.mark.reporter
@pytest.mark.unit
class TestReporterVisualizationFeatures:
    """Test visualization capabilities: cosine similarity and t-SNE."""
    
    def test_cosine_similarity_matrix_generation(self, reporter, tmp_path):
        """Test cosine similarity matrix computation and saving."""
        reporter.output_dir_result = tmp_path
        
        # Create results with embeddings
        results_with_embeddings = []
        for i in range(4):
            result = {
                "bbox": (10*i, 20*i, 50, 60),
                "score": 0.9,
                "embedding": np.random.rand(128).astype(np.float32),
                "label": f"Person{i}"
            }
            results_with_embeddings.append(result)
        
        img_path = tmp_path / "similarity_test.jpg"
        
        reporter.save_cosine_similarity_matrix(img_path, results_with_embeddings)
        
        # Verify files were created
        similarity_file = tmp_path / "cosine_similarity_matrix_similarity_test.csv"
        assert similarity_file.exists()
        
        # Verify matrix structure
        df = pd.read_csv(similarity_file)
        assert df.shape[0] == df.shape[1] == 4  # Square matrix
        
        # Values should be between 0 and 1 (cosine similarity)
        assert df.min().min() >= 0.0
        assert df.max().max() <= 1.0
    
    def test_cosine_similarity_insufficient_data(self, reporter, tmp_path):
        """Test cosine similarity with insufficient embeddings."""
        reporter.output_dir_result = tmp_path
        
        # Only one embedding - should handle gracefully
        single_result = [{
            "bbox": (10, 20, 50, 60),
            "embedding": np.random.rand(128).astype(np.float32)
        }]
        
        img_path = tmp_path / "insufficient.jpg"
        reporter.save_cosine_similarity_matrix(img_path, single_result)
        
        # No similarity file should be created
        similarity_files = list(tmp_path.glob("*similarity*.csv"))
        assert len(similarity_files) == 0
    
    def test_tsne_visualization(self, reporter, tmp_path):
        """Test t-SNE visualization generation."""
        # Create sufficient data for t-SNE
        test_data = []
        for i in range(15):  # Need enough points for t-SNE
            test_data.append({
                "embedding": np.random.rand(128).astype(np.float32),
                "label": f"Person{i % 5}"
            })
        
        reporter.output_dir = tmp_path
        path = tmp_path / "tsne_test"
        
        reporter.save_tsne_visualization(path, test_data)
        
        # Verify t-SNE file was created
        tsne_file = tmp_path / "tsne_visualization_tsne_test.svg"
        assert tsne_file.exists()
        assert tsne_file.stat().st_size > 0


@pytest.mark.reporter
@pytest.mark.unit
class TestReporterErrorHandling:
    """Test error handling and edge cases."""
    
    def test_save_without_output_directory(self, tmp_path):
        """Test error when saving without proper output directory setup."""
        config = ReporterConfig(output_dir=tmp_path)
        reporter = Reporter(config)
        
        # Trying to save without setup should raise error
        with pytest.raises(RuntimeError, match="Output directory structure not created"):
            reporter.save_to_file(Path("test.json"), [{"test": "data"}])
    
    def test_malformed_results_handling(self, reporter, tmp_path):
        """Test handling of malformed or incomplete results data."""
        reporter.output_dir_result = tmp_path
        
        # Results with missing keys or malformed data
        malformed_results = [
            {"bbox": (10, 20, 30, 40)},  # Missing other expected keys
            {"score": 0.5},              # Missing bbox
            {"embedding": "not_an_array"},  # Invalid embedding type
            {}                           # Empty result
        ]
        
        # Should handle malformed data gracefully
        reporter.save_to_file(
            tmp_path / "malformed.json", 
            malformed_results, 
            format=OutputFormat.JSON
        )
        
        assert (tmp_path / "malformed.json").exists()
    
    def test_reporter_with_saving_disabled(self, tmp_path, dummy_image, dummy_results):
        """Test Reporter behavior when saving is completely disabled."""
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
            def settings(self): return {}
        
        img_path = tmp_path / "disabled_test.jpg"
        cv2.imwrite(str(img_path), dummy_image)
        
        # Should exit early and not create any files
        reporter.save_processed_image_results(
            MockModel(), MockModel(), MockModel(),
            dummy_image.copy(), img_path,
            results=dummy_results.copy()
        )
        
        # No output files should be created
        assert not list(tmp_path.rglob("*_results.*"))
        assert not list(tmp_path.rglob("*_annotated.*"))


# Legacy compatibility tests (simplified)
def test_legacy_save_processed_image_results(reporter, dummy_image, dummy_results, tmp_path):
    """Legacy compatibility test for save_processed_image_results."""
    class Dummy:
        def settings(self): return {"param": "value"}
    
    detector = embedder = classifier = Dummy()
    img_path = tmp_path / "legacy_test.jpg"
    cv2.imwrite(str(img_path), dummy_image)

    reporter.save_processed_image_results(
        detector, embedder, classifier,
        dummy_image.copy(), img_path,
        results=dummy_results.copy(),
        cropped_faces=[dummy_image.copy(), dummy_image.copy()]
    )

    # Basic validation
    outdir = reporter.output_dir_result
    assert outdir.exists()
    assert (outdir / "model_settings.json").exists()


def test_legacy_bulk_mode(reporter, dummy_image, dummy_results, tmp_path):
    """Legacy compatibility test for bulk mode."""
    reporter.setup_bulk_output_directory()
    
    class Dummy:
        def settings(self): return {"param": "value"}
    
    detector = embedder = classifier = Dummy()
    img_path = tmp_path / "bulk_legacy.jpg"
    cv2.imwrite(str(img_path), dummy_image)

    reporter.save_processed_image_results(
        detector, embedder, classifier,
        dummy_image.copy(), img_path,
        results=dummy_results.copy()
    )

    # Verify bulk structure
    img_subdir = reporter.output_dir / "bulk_legacy"
    assert img_subdir.exists()