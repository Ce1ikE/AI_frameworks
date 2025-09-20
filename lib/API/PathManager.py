from pathlib import Path
from menu import Menu
import logging


class PathManager:
    logger = logging.getLogger(__name__)

    def __init__(self, entrypoint: str):
        try:
            self.m_entrypoint_dir = Path(entrypoint).resolve().parent
        except Exception as e:
            message = f"Failed to resolve entrypoint directory: {e}"
            self.logger.error(message)
            raise RuntimeError(message)
        
        self.dataset = None
        self.input = None
        self.output = None
        self.logs = None
        self.models = None
    
    def initialize(self, config):
        try:
            self.run_health_checks(config)
            self.logger.info("All required directories are present.")
        except Exception as e:
            message = f"Initialization failed: {e}"
            self.logger.error(message)
            raise RuntimeError(message)

    def run_health_checks(self, config):        
        for rel_path in config["settings"].values():
            if not Path(self.m_entrypoint_dir / rel_path).is_dir():
                message = f"Required directory not found: {rel_path}"
                self.logger.error(message)
                raise FileNotFoundError(message)

        try:
            self.dataset = Path(self.m_entrypoint_dir / config["settings"]["dataset_dir"])
            self.input   = Path(self.m_entrypoint_dir / config["settings"]["input_dir"])
            self.output  = Path(self.m_entrypoint_dir / config["settings"]["output_dir"])
            self.logs    = Path(self.m_entrypoint_dir / config["settings"]["log_dir"])
            self.models  = Path(self.m_entrypoint_dir / config["settings"]["model_dir"])
        except KeyError as e:
            message = f"Missing configuration key: {e}"
            self.logger.error(message)
            raise ValueError(message)
