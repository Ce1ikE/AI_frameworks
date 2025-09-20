from pathlib import Path
from lib.API.PathManager import PathManager
from lib.API.Reporter import Reporter
from lib.API.Preprocessor import Preprocessor
from lib.API.FaceDetector import FaceDetector
from lib.API.FaceEmbedder import FaceEmbedder
from lib.API.Pipeline import Pipeline
from lib.Core import Core 


def main():

    path_manager = PathManager(entrypoint=__file__)

    pipeline=Pipeline(
        face_detector=FaceDetector(),
        face_embedder=FaceEmbedder(),
        preprocessor=Preprocessor(),
        reporter=Reporter()
    )

    Core(path_manager).run(pipeline,image_path=Path("path/to/your/image.jpg"))

if __name__ == "__main__":
    main()

