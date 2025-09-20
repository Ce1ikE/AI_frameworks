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
    
    def __init__(self, path_manager: PathManager):
        self.paths = path_manager
        self.m_args = self.parse_arguments()
        self.m_config = self.parse_config()
        self.setup_logging()
        self.set_log_level()

    def parse_arguments(self):
        # https://docs.python.org/3/howto/argparse.html
        parser = argparse.ArgumentParser()
        parser.add_argument("-v","--verbose", action="store_true", help="Enable verbose logging")
        return parser.parse_args()
    
    def parse_config(self):
        config_file = self.paths.entrypoint_dir / "config.toml"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with config_file.open("rb") as f:
            return tomllib.load(f)
        
    def setup_logging(self):
        log_file = Path(self.paths.logs / f"script_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        # https://docs.python.org/3/library/logging.html
        # there shouldn't be duplicate log files (because of datetime) but just in case, we use mode="w"
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(levelname)s][%(asctime)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file,mode="w"),
            ]
        )
        self.logger.debug(f"Logging initialized. Log file: {log_file}")
    
    def set_log_level(self):
        if self.m_args.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def run(self, pipeline: Pipeline, image_path: Path = None):
        pipeline.process(image_path)
