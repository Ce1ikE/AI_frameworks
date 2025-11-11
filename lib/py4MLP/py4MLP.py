# py4MLP ---------------------------------------------------------------------
#                           /$$   /$$ /$$      /$$ /$$       /$$$$$$$ 
#                          | $$  | $$| $$$    /$$$| $$      | $$__  $$
#        /$$$$$$  /$$   /$$| $$  | $$| $$$$  /$$$$| $$      | $$  \ $$
#       /$$__  $$| $$  | $$| $$$$$$$$| $$ $$/$$ $$| $$      | $$$$$$$/
#      | $$  \ $$| $$  | $$|_____  $$| $$  $$$| $$| $$      | $$____/ 
#      | $$  | $$| $$  | $$      | $$| $$\  $ | $$| $$      | $$      
#      | $$$$$$$/|  $$$$$$$      | $$| $$ \/  | $$| $$$$$$$$| $$      
#      | $$____/  \____  $$      |__/|__/     |__/|________/|__/      
#      | $$       /$$  | $$                                           
#      | $$      |  $$$$$$/                                           
#      |__/       \______/           
# ----------------------------------------------------------------------------

PIPELINE_DIR = "py4MLP_pipelines"
DEFAULT_CONFIG_FILE = "py4MLP_config.toml"

from pathlib import Path

class PathsConfig:
    def __init__(self):
        self.output = Path()
        self.logs = Path()
        self.models = Path()
 
class Py4MLP:
    """
    Core class to initialize the pipeline library components.
    """
    @classmethod
    def __init__(cls, enable_logging: bool = True):
        import os
        cls.entrypoint = os.getcwd()
        cls.paths = PathsConfig()
        cls.parse_arguments()
        cls.parse_config()
        cls.setup_paths()
        if enable_logging:
            cls.setup_logging()
    
    @classmethod
    def parse_config(cls):
        import tomllib
        pass
    
    @classmethod
    def parse_arguments(cls):
        import argparse
        pass

    @classmethod
    def setup_logging(cls):
        import sys
        import logging
        import datetime as dt 
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(Py4MLP.paths.logs / f"log__{timestamp}.log")
        logging.basicConfig(
            force=True,
            level=logging.INFO,
            format='[%(levelname)s] - [%(asctime)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file,mode="w"),
                logging.StreamHandler(stream=sys.stdout)
            ]
        )
        cls.logger = logging.getLogger(__name__)
        cls.logger.debug(f"Logging initialized. Log file: {log_file}")

    @classmethod
    def setup_paths(cls):
        # preferably we want to setup the output directory outside the the cwd
        # so we move one up and   
        cls.entrypoint_dir = Path(cls.entrypoint).resolve().parent
        cls.pipeline_library_dir = cls.entrypoint_dir / PIPELINE_DIR
        cls.pipeline_library_dir.mkdir(parents=True, exist_ok=True)

        try:
            cls.paths.output  = Path(cls.pipeline_library_dir / "results")
            cls.paths.logs    = Path(cls.pipeline_library_dir / "logs")
            cls.paths.models  = Path(cls.pipeline_library_dir / "models")

            for p in [cls.paths.output, cls.paths.logs, cls.paths.models]:
                p = p.resolve()
                p.mkdir(parents=True, exist_ok=True)
        except KeyError as e:
            raise ValueError(f"Missing configuration key: {e}")
