# AI Frameworks Test Suite

This directory contains comprehensive unit and integration tests for the AI Frameworks pipeline system.

## Test Structure

The test suite is organized into several categories:

### Pipeline Tests (`test_pipeline.py`)
- **TestPipelineComponentValidation**: Tests for component validation logic
- **TestPipelineDataValidation**: Tests for data validation and error handling
- **TestPipelineImageProcessing**: Tests for image processing methods
- **TestPipelineProcessMethod**: Tests for single image processing
- **TestPipelineBulkProcessMethod**: Tests for batch processing
- **TestPipelineTrainMethod**: Tests for unsupervised training functionality

### Reporter Tests (`test_reporter.py`)
- **TestReporterConfig**: Tests for ReporterConfig dataclass and configuration validation
- **TestReporterInitialization**: Tests for Reporter class initialization
- **TestReporterDirectoryManagement**: Tests for directory setup and management
- **TestReporterDataValidation**: Tests for input validation and edge cases
- **TestReporterFileSaving**: Tests for file saving in multiple formats (JSON, CSV, TXT, BIN, Parquet)
- **TestReporterVisualizationMethods**: Tests for cosine similarity matrices and t-SNE visualization
- **TestReporterBulkOperations**: Tests for bulk processing and results compilation
- **TestReporterTrainingWorkflow**: Tests for training-related functionality
- **TestReporterAnnotationMethods**: Tests for image annotation and processing
- **TestReporterIntegration**: End-to-end Reporter workflow tests
- **TestReporterErrorHandling**: Tests for error scenarios and edge cases

### Integration Tests  
- **TestPipelineIntegration**: End-to-end tests with real components

## Test Markers

Tests are categorized using pytest markers:

- `unit`: Unit tests that test individual functions/methods in isolation
- `integration`: Integration tests that test multiple components working together
- `slow`: Tests that take a long time to run
- `gpu`: Tests that require GPU resources  
- `network`: Tests that require network access
- `real_models`: Tests that use real AI models (not mocks)
- `reporter`: Tests specific to the Reporter class functionality
- `config`: Tests for configuration and settings validation
- `visualization`: Tests for visualization and plotting features

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Run with verbose output
pytest -v

# Run specific test file
pytest test_pipeline.py
pytest test_reporter.py

# Run specific test class
pytest test_pipeline.py::TestPipelineComponentValidation
pytest test_reporter.py::TestReporterConfig

# Run specific test method
pytest test_pipeline.py::TestPipelineComponentValidation::test_validate_components_for_processing_missing_detector
pytest test_reporter.py::TestReporterConfig::test_default_config_creation

# Run Reporter-specific tests
pytest -m reporter

# Run configuration tests
pytest -m config

# Run visualization tests (may be slow)
pytest -m visualization
```

### Using the Test Runner Script

```bash
# Run all tests with verbose output
python tests/run_tests.py --verbose

# Run only unit tests
python tests/run_tests.py --unit

# Run only integration tests
python tests/run_tests.py --integration

# Run with coverage report
python tests/run_tests.py --coverage

# Run tests matching a pattern
python tests/run_tests.py --pattern "validation"

# Run specific test file
python tests/run_tests.py --file test_pipeline.py

# Show available test markers
python tests/run_tests.py --markers
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=lib --cov-report=html

# Generate terminal coverage report
pytest --cov=lib --cov-report=term

# Generate both HTML and terminal reports
pytest --cov=lib --cov-report=html --cov-report=term
```

## Test Data and Fixtures

### Available Fixtures

- `reporter`: A configured Reporter instance for testing
- `mock_detector`: Mock face detector for unit tests
- `mock_embedder`: Mock face embedder for unit tests  
- `mock_classifier`: Mock face classifier for unit tests
- `mock_reporter`: Mock reporter for unit tests
- `pipeline_minimal`: Minimal pipeline with detector and reporter
- `pipeline_full`: Full pipeline with all mocked components
- `pipeline_full_real`: Pipeline with real components for integration testing
- `dummy_image_array`: Random image array for testing
- `dummy_image_path`: Temporary image file for testing
- `multiple_dummy_images`: Multiple temporary image files for batch testing
- `dummy_results`: Sample processing results for testing
- `dummy_image`: Dummy image array (480x640x3) with random pixel values

### Test Images

The test suite uses images from the AFW (Annotated Faces in the Wild) dataset located in `tests/test_images/`. If no real images are available, dummy images are generated automatically.

## Reporter Test Coverage

The Reporter test suite provides comprehensive coverage of all output and visualization functionality:

### Configuration Testing
- ReporterConfig dataclass validation
- Default and custom configuration settings
- Property validation and is_saving_enabled logic

### Directory Management
- Single image output directory setup
- Bulk processing directory structure
- Training output directory organization
- Directory path resolution and creation

### File Format Support
- JSON output format
- CSV output format  
- TXT output format
- Binary (NPZ) output format
- Parquet output format
- Error handling for unsupported formats

### Visualization Features
- Cosine similarity matrix computation and saving
- t-SNE visualization generation
- Visualization error handling with insufficient data
- SVG and HTML format support

### Integration Workflows
- Complete single image processing workflow
- Bulk processing with results compilation
- Training workflow with model saving
- Error handling and recovery scenarios

### Demo Scripts
Run `python tests/demo_reporter_tests.py` to see live examples of Reporter functionality.

## Best Practices

### Writing Unit Tests

1. **Use mocks for external dependencies**: Mock file I/O, model inference, etc.
2. **Test one thing at a time**: Each test should verify a single behavior
3. **Use descriptive test names**: Names should clearly indicate what is being tested
4. **Test edge cases**: Include tests for error conditions and boundary cases
5. **Keep tests fast**: Unit tests should run quickly

### Writing Integration Tests

1. **Test realistic scenarios**: Use real components and data when possible
2. **Mark slow tests appropriately**: Use `@pytest.mark.slow` for long-running tests
3. **Clean up resources**: Ensure temporary files and directories are cleaned up
4. **Test error recovery**: Verify the system handles failures gracefully

### Test Data Management

1. **Use fixtures for common test data**: Avoid duplicating test setup code
2. **Keep test data small**: Use minimal data sets that still test the functionality
3. **Generate data programmatically**: Prefer generated test data over static files when possible
4. **Clean up after tests**: Remove temporary files and reset state

## CI/CD Integration

The test suite is designed to work with continuous integration systems. Key considerations:

- Tests are hermetic (no external dependencies beyond the test environment)
- Temporary directories are properly cleaned up
- Tests can run in parallel where appropriate
- Coverage reports are generated in standard formats

## Debugging Tests

### Common Issues

1. **Import errors**: Ensure the `lib` package is in the Python path
2. **Missing test data**: Check that test images exist in `tests/test_images/`
3. **Permission errors**: Ensure test output directories are writable
4. **Model loading failures**: Integration tests may fail if model files are missing

### Debugging Commands

```bash
# Run tests with maximum verbosity
pytest -vvv -s

# Run specific failing test in debug mode
pytest -vvv -s test_pipeline.py::TestClass::test_method

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest --tb=long
```

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Add appropriate pytest markers
3. Include docstrings explaining what the test does
4. Update this README if adding new test categories
5. Ensure tests pass in isolation and as part of the full suite

## Dependencies

The test suite requires:

- pytest >= 7.0
- pytest-cov (for coverage reports)
- numpy
- pandas
- opencv-python
- All dependencies of the main AI Frameworks package