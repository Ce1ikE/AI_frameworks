# AI Frameworks 🤖

## project: setup
> `lib/Py4MLP` contains the custom pipeline library to run the pipeline plugin(s) 

> `pipeline_plugin` implementation for face detection, recognition and classification using the Py4MLP library 

> `wheels` prebuild distribution of pillow-heif such that pillow supports reading .HEIC image formats  

> `main.py` entrypoint to the application

## project: how to run ?
First you must install the required libraries either using the `requirements.txt` or the `pyproject.toml`
```sh
# create a virtual environment either with venv
python -m venv ./venv
# activate the virtual environment
./venv/scripts/activate
# and use pip
pip install -r requirements.txt

# or with uv
uv add -r requirements.txt
```

after you can just modify and run the `main.py` , which is the entrypoint of this application

```sh
python main.py 
# or with uv
uv run main.py
```


> **NOTE**
most of these models will not be available in this REPO due to file size that is why implementing a model store from which you can download the model is better this avoids cluttering the library the model store functions are the ones from the uniface REPO https://github.com/yakhyo but tweeked slightly to allow all models to use these function.

## Running tests

This project uses pytest and includes lightweight unit tests for both the library and the application.

Quick steps (PowerShell / pwsh):

```powershell
# create and activate a virtual environment
python -m venv ./.venv
# PowerShell activation
. .\.venv\Scripts\Activate.ps1

# install dependencies (and pytest)
pip install -r requirements.txt pytest

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
