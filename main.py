from pathlib import Path
from lib.API.PathManager import PathManager
from lib.API.Reporter import Reporter , OutputFormat
from lib.API.Preprocessor import Preprocessor
from lib.face_detection.ViolaJones import ViolaJones
from lib.face_detection.CascadeType import CascadeType
from lib.API.FaceEmbedder import FaceEmbedder
from lib.API.Pipeline import Pipeline
from lib.Core import Core 


def main():

    path_manager = PathManager(entrypoint=__file__)
    core = Core(path_manager)
    
    path_manager.resolve_dirs(core.m_config)

    pipeline = Pipeline(
        face_detector=ViolaJones(
            cascade_type=CascadeType.FRONTALFACE_DEFAULT
        ),
        face_embedder=None,
        preprocessor=Preprocessor(
            target_size=(160,160)
        ),
        reporter=Reporter(
            output_dir=path_manager.output,
            save_to_file=True,
            save_format=OutputFormat.CSV,
            save_cropped_faces=True,
            save_preprocessed_faces=True,
        )
    )

    core.run(pipeline)

if __name__ == "__main__":
    main()

