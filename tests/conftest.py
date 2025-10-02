import numpy as np
import pytest
import cv2
from pathlib import Path
import numpy as np
import time
import shutil
from numpy.random import default_rng
from lib.API.Preprocessor import Preprocessor
from lib.API.Reporter import Reporter , ReporterConfig

# https://pythontest.com/pytest-tips-tricks/
# https://docs.pytest.org/en/6.2.x/tmpdir.html#base-temporary-directory

@pytest.fixture
def reporter():
    base_output_dir: Path = Path(__file__).parent / "test_output_dir"
    
    dir_name = "test_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(int(time.time() * 1000) % 1000)
    base_output_dir = base_output_dir / dir_name
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ReporterConfig(output_dir=base_output_dir)
    return Reporter(config)

@pytest.fixture
def preprocessor():
    return Preprocessor(target_size=(112, 112))

@pytest.fixture
def dummy_image():
    generator = default_rng(42)
    # load a test image from the test_images directory 
    # [AFW (Annotated Faces in the Wild) dataset (https://exposing.ai/afw/)]
    dir_with_test_images = Path(__file__).parent.absolute() / "test_images"
    # take a random image from the directory
    image_files = list(dir_with_test_images.glob("*.jpg")) + list(dir_with_test_images.glob("*.png"))
    if image_files:
        img_path = generator.choice(image_files)
        # read the actual image and return the array
        image = cv2.imread(str(img_path))
        if image is not None:
            return image
    # if no image found, create a dummy image (480x640 with 3 channels RGB with random values)
    return generator.integers(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def dummy_results():
    return [
        {
            "bbox": (10, 20, 50, 60),
            "score": 0.95,
            "embedding": np.random.rand(128).astype(np.float32),
            "label": "PersonA"
        },
        {
            "bbox": (70, 30, 40, 40),
            "score": 0.88,
            "embedding": np.random.rand(128).astype(np.float32),
            "label": "PersonB"
        },
        {
            "bbox": (15, 25, 30, 30), 
            "score": 0.80, 
            "embedding": np.random.rand(128).astype(np.float32), 
            "label": "PersonC"
        },
        {
            "bbox": (60, 20, 25, 25), 
            "score": 0.75, 
            "embedding": np.random.rand(128).astype(np.float32), 
            "label": "PersonD"
        }
    ]