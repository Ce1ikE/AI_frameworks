# std packages
import argparse
import logging
import tomllib
from datetime import datetime
from pathlib import Path
from menu import Menu
from os.path import dirname
# https://medium.com/pythons-gurus/what-is-the-best-face-detector-ab650d8c1225

from .API.PathManager import PathManager 
from .API.Pipeline import (Pipeline,Preprocessor,FaceDetector,FaceEmbedder,Reporter)

class Core:
    logger = logging.getLogger(__name__)
    
    def __init__(
        self, 
        path_manager: PathManager
    ):
        self.paths = path_manager

        self.m_args = self.parse_arguments()
        self.m_config = self.parse_config()

        self.paths.initialize(self.m_config)
        
        self.setup_logging()
        self.set_log_level()


    def setup_logging(self):
        path_to_log_file = Path(self.paths.logs / f"script_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # https://docs.python.org/3/library/logging.html
        # there shouldn't be duplicate log files (because of datetime) but just in case, we use mode="w"
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(levelname)s][%(asctime)s] - %(message)s',
            handlers=[
                logging.FileHandler(path_to_log_file.as_posix(),mode="w"),
            ]
        )
        self.logger.debug(f"Logging initialized. Log file: {path_to_log_file}")
    
    def parse_config(self):
        try:
            config_file = self.paths.m_entrypoint_dir / "config.toml"
            config_path = Path(config_file).resolve()
        except:
            raise FileNotFoundError("Configuration file 'config.toml' not found. Please ensure it exists in the script directory.")
        # https://docs.python.org/3/library/tomllib.html#module-tomllib
        with config_path.open("rb") as f:
            config = tomllib.load(f)
        return config
    
    def parse_arguments(self):
        # https://docs.python.org/3/howto/argparse.html
        parser = argparse.ArgumentParser()
        parser.add_argument("-v","--verbose", action="store_true", help="Enable verbose logging")
        args = parser.parse_args()
        return args
    
    def set_log_level(self):
        if self.m_args.verbose:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("Verbose mode enabled")
        else:
            self.logger.setLevel(logging.INFO)
            self.logger.info("Verbose mode disabled")

    def run(
        self,
        pipeline: Pipeline, 
        image_path: Path = None
    ):
        pipeline.process(image_path)

    # def show_main_menu(self):
    #     menu = Menu(title="AI Frameworks - Main Menu")
    #     menu.set_options(
    #         options=(
    #             ("Viola-Jones Face Detection", lambda: run_viola_jones_face_detection(self.paths)),
    #             ("CNN Face Detection", lambda: run_cnn_face_detection(self.paths)),
                
    #             ("CNN Face Identification", lambda: run_cnn_face_representation(self.paths)),

    #             ("Knn Face Classification", lambda: run_knn_face_classification(self.paths)),
    #             ("SVM Face Classification", lambda: run_svm_face_classification(self.paths)),
    #             ("Exit",menu.close)
    #         )
    #     )
    #     menu.open()