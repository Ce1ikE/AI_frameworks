import sys
import argparse
import logging
import tomllib
from datetime import datetime
from pathlib import Path
from .API.PathManager import PathManager


class Core:
    logger = logging.getLogger(__name__)
    
    def __init__(self, entrypoint: str, config: str = "config.toml"):
        self.paths = PathManager(entrypoint, config)
        self.m_args = self.parse_arguments()
        self.m_config = self.parse_config()
        self.paths.resolve_dirs(self.m_config)

        self.setup_logging()
        self.set_log_level()

    def parse_arguments(self):
        # https://docs.python.org/3/howto/argparse.html
        parser = argparse.ArgumentParser()
        parser.add_argument("-v","--verbose", action="store_true", help="Enable verbose logging")
        return parser.parse_args()
    
    def parse_config(self):
        config_file = self.paths.config
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with config_file.open("rb") as f:
            return tomllib.load(f)
        
    def setup_logging(self):
        log_file = Path(self.paths.logs / f"script_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        # https://docs.python.org/3/library/logging.html
        # there shouldn't be duplicate log files (because of datetime) but just in case, we use mode="w"
        logging.basicConfig(
            force=True,
            level=logging.DEBUG,
            format='[%(levelname)s][%(asctime)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file,mode="w"),
                logging.StreamHandler(stream=sys.stdout)
            ]
        )
        self.logger.debug(f"Logging initialized. Log file: {log_file}")

    def set_log_level(self):
        if self.m_args.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.debug(f"Log level set to {self.logger.level}")



