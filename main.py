from zipfile import Path
from lib.API.PathManager import PathManager
from lib.API.Pipeline import Pipeline
from lib.API.Reporter import Reporter , OutputFormat

from lib.face_detection.ViolaJones import ViolaJonesDetector , CascadeType
from lib.face_detection.RetinaFace import RetinaFaceDetector
from lib.face_detection.SCRFD import SCRFDDetector
from lib.face_detection.HoG import HoGDetector
from lib.face_detection.YuNet import YuNetDetector

from lib.face_representation.ArcFace import ArcFaceEmbedder

from uniface.constants import RetinaFaceWeights

from lib.Core import Core 

def main():

    path_manager = PathManager(entrypoint=__file__, config="config.toml")
    core = Core(path_manager)
    
    input_files = path_manager.input.glob("*.jpg")
    
    for input_image_path in input_files:
        Pipeline(
            detector=RetinaFaceDetector(
                model_path=RetinaFaceWeights.MNET_025,
                confidence_threshold=0.6
            ),
            embedder=ArcFaceEmbedder(
                model_path=path_manager.models / "arcface" / "w600k_mbf.onnx"
            ),
            reporter=Reporter(
                output_dir=path_manager.output,
                save_original_image=True,
                save_to_file=True,
                save_format=OutputFormat.CSV,
                save_cropped_faces=True,
            )
        ).run(input_image_path.resolve())


if __name__ == "__main__":
    main()

