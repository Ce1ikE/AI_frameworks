from pathlib import Path
import logging
from menu import Menu
import pprint
from ..API.PathManager import PathManager
from .CascadeType import CascadeType
from .ViolaJones import ViolaJones


def run_viola_jones_face_detection__default(paths: PathManager):
    vj_detector = ViolaJones(
        input_image=paths.input / "group_photo.jpg",
        output_image_dir=paths.output,
        cascade_type=CascadeType.FRONTALFACE_DEFAULT
    )
    vj_detector.detect_and_draw_faces()
    vj_detector.save_output_image()

def run_viola_jones_face_detection__pipeline(paths: PathManager):
    files_in_input = paths.list_files(paths.input)
    
    print("Running Viola Jones over: ")
    pprint.pprint(files_in_input)
    
    for file in files_in_input:
        print(f"Running on : {file}")
        vj_detector = ViolaJones(
            input_image=paths.input / file,
            output_image_dir=paths.output,
            cascade_type=CascadeType.FRONTALFACE_DEFAULT
        )
        vj_detector.detect_and_draw_faces()
        vj_detector.save_output_image()
        vj_detector.save_output_faces(save_in_sub_dir=True)

def run_viola_jones_face_detection(paths: PathManager):
    menu = Menu(title = "Viola-Jones Face Detection")
    menu.set_options(
        options=(
            ("Run with default settings", lambda: run_viola_jones_face_detection__default(paths)),
            ("Run pipeline", lambda: run_viola_jones_face_detection__pipeline(paths)),
            ("Back to Main Menu", menu.close)
        )
    )
    menu.open()

def run_cnn_face_detection__default(paths: PathManager):
    pass

def run_cnn_face_detection__pipeline(paths: PathManager):
    pass

def run_cnn_face_detection(paths: PathManager):
    menu = Menu(title = "CNN Face Detection")
    menu.set_options(
        options=(
            ("Run with default settings", lambda: run_cnn_face_detection__default(paths)),
            ("Run pipeline", lambda: run_cnn_face_detection__pipeline(paths)),
            ("Back to Main Menu", menu.close)
        )
    )
    menu.open()