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
# -v for verbose output
python main.py -v 
```


> **NOTE**
most of these models will not be available in this REPO due to file size that is why implementing a model store from which you can download the model is better this avoids cluttering the library the model store functions are the ones from the uniface REPO https://github.com/yakhyo but tweeked slightly to allow all models to use these function.