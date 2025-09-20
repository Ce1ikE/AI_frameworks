from pathlib import Path
import logging
from menu import Menu
from ..API.PathManager import PathManager

def run_cnn_face_representation__default(paths: PathManager):
    pass

def run_cnn_face_representation__pipeline(paths: PathManager):
    pass

def run_cnn_face_representation(paths: PathManager):
    menu = Menu(title = "CNN Face representation")
    menu.set_options(
        options=(
            ("Back to Main Menu", menu.close)
        )
    )
    menu.open()