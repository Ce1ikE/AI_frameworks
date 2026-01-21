# AI Frameworks 🤖

## project: setup
> `lib/Py4MLP` contains the custom pipeline library to run the pipeline plugin(s) 

> `pipeline_plugin` implementation for face detection, recognition and classification using the Py4MLP library 

> `wheels` prebuild distribution of pillow-heif such that pillow supports reading .HEIC image formats  

> `main.py` entrypoint to the application

## project: how to run ?
First you must install the required libraries using the `pyproject.toml`:

```sh
# create a virtual environment with venv
python -m venv ./.venv
# activate the virtual environment
./.venv/scripts/activate

# or with uv (recommended)
uv venv
# activate the virtual environment
./.venv/scripts/activate
uv sync
```

After installation, you can run the `main.py` entrypoint with various command-line options:

```sh
# Run with default settings (no pipelines will execute)
python main.py

# or with uv
uv run main.py

# Run feature extraction pipeline
python main.py --run-feature-extraction

# Run training pipeline with custom cluster count
python main.py --run-training --n-clusters 12

# Run inference pipeline
python main.py --run-inference

# Run streaming pipeline
python main.py --run-streaming

# Combine multiple pipelines
python main.py --run-feature-extraction --run-training --run-inference

# Custom directories
python main.py --train-dir ./my_dataset/train --test-dir ./my_dataset/test --output-dir ./results

# Enable logging
python main.py --run-feature-extraction --enable-logging
```

### Command-line arguments:
- `--train-dir`: Directory containing training images (default: `./dataset/train`)
- `--test-dir`: Directory containing test images (default: `./dataset/test`)
- `--output-dir`: Output directory for results (default: uses Py4MLP output path)
- `--n-clusters`: Number of clusters for training pipeline (default: 13)
- `--enable-logging`: Enable Py4MLP logging
- `--run-feature-extraction`: Run feature extraction pipeline
- `--run-training`: Run training pipeline
- `--run-inference`: Run inference pipeline
- `--run-streaming`: Run streaming pipeline
- `--model-info`: Path to model info parquet file (default: `./pipeline_plugin/model_info.parquet`)

> **NOTE**
Most of these models will not be available in this REPO due to file size. The model store implementation downloads models as needed, avoiding repository clutter. The model store functions are adapted from the uniface REPO https://github.com/yakhyo.

## Running tests

This project uses pytest and includes lightweight unit tests for both the library and the application.

Quick steps (PowerShell / pwsh):

```powershell
# create and activate a virtual environment
python -m venv ./.venv
# PowerShell activation
. .\.venv\Scripts\Activate.ps1

# install dependencies (pytest is included in pyproject.toml)
uv sync
# or with pip
pip install -e .

# run the full test suite (quiet)
python -m pytest -q

# run tests in a single folder
python -m pytest -q tests/lib
python -m pytest -q tests/app

# run a single test file
python -m pytest -q tests/app/test_main.py
```

Notes:
- Tests live under the `tests/` directory (`tests/lib` and `tests/app`).
- Some tests stub heavy third-party modules to keep the test run lightweight and avoid downloading large models at import time.
- If you prefer to run tests with verbosity or see captured logs, omit `-q` or use `-k` to filter by test name.
