from pathlib import Path
import logging
from menu import Menu
from ..API.PathManager import PathManager


def run_knn_face_classification__default(paths: PathManager):
    pass

def run_knn_face_classification__pipeline(paths: PathManager):
    pass

def run_knn_face_classification(paths: PathManager):
    menu = Menu(title = "Knn Face classification")
    menu.set_options(
        options=(
            ("Back to Main Menu", menu.close)
        )
    )
    menu.open()

def run_svm_face_classification__default(paths: PathManager):
    pass

def run_svm_face_classification__pipeline(paths: PathManager):
    pass

def run_svm_face_classification(paths: PathManager):
    menu = Menu(title = "SVM Face classification")
    menu.set_options(
        options=(
            ("Back to Main Menu", menu.close)
        )
    )
    menu.open()