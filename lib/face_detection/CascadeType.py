import os
from enum import Enum
from pathlib import Path
file_path = Path(__file__).resolve().parent
HAARCASCADE_PATH = file_path / "haarcascades"

class CascadeType(Enum):   
    FRONTALFACE_DEFAULT = HAARCASCADE_PATH / "haarcascade_frontalface_default.xml"
    FRONTALFACE_ALT = HAARCASCADE_PATH / "haarcascade_frontalface_alt.xml"
    FULLBODY = HAARCASCADE_PATH / "haarcascade_fullbody.xml"

    __all__ = ["FRONTALFACE_DEFAULT", "FRONTALFACE_ALT", "FULLBODY"]
        
    