from lib.API.Pipeline import Pipeline
from lib.API.Reporter import Reporter , OutputFormat

# detectors (img -> bboxes + optionally landmarks + scores)
from lib.face_detection.ViolaJones import ViolaJonesDetector , CascadeType
from lib.face_detection.RetinaFace import RetinaFaceDetector
from lib.face_detection.YuNet import YuNetDetector
from lib.face_detection.SCRFD import SCRFDDetector
from lib.face_detection.HoG import HoGDetector

# embedders (face img -> embedding vector)
from lib.face_representation.ArcFace import ArcFaceEmbedder , ArcFaceWeights

from uniface.constants import RetinaFaceWeights

from lib.Core import Core 
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pprint


# main is the entrypoint of the application
# it sets up the PathManager for assuring that everything is in place, 
# Core sets up the necessary components like logging parsing config files and arguments 
# the Pipeline is where the actual work is done where the detector, embedder and reporter are used
# the detector detects faces in an image, the embedder creates embeddings for those faces
# the reporter saves the results to the output directory
# TODO: add a ML model for classification or clustering of the embeddings
# TODO: add a option to the reporter to save a PDF report
# TODO: add a option to the reporter to save a HTML report
# TODO: add training of a classifier on the embeddings to the Pipeline
# TODO: add a option to the reporter to save the classifier model
# TODO: add a module to load a classifier model and use it in the Pipeline
# TODO: add a option to the reporter to save the visualization of the embeddings
# TODO: add a option to the reporter to save the visualization of the clusters



def main():

    core = Core(entrypoint=__file__)

    input_files = list(core.paths.input.glob("*.jpg")) + list(core.paths.input.glob("*.png")) + list(core.paths.input.glob("*.jpeg"))
    core.logger.info(f"Found {len(input_files)} input files.")
    core.logger.info(f"Input files: {pprint.pformat(input_files)}")

    for input_image_path in input_files:
        Pipeline(
            bulk_mode=True,
            detector=RetinaFaceDetector(
                model_name=RetinaFaceWeights.MNET_025,
                confidence_threshold=0.6
            ),
            embedder=ArcFaceEmbedder(
                model_name=ArcFaceWeights.W600K_MBF
            ),
            classifier=None,
            reporter=Reporter(
                output_dir=core.paths.output,
                prefix_save_dir="ver1",
                save_original_image=True,
                save_to_file=True,
                save_format=OutputFormat.CSV,
                save_cropped_faces=True,
            )
        ).run(input_image_path.resolve())


if __name__ == "__main__":
    main()

