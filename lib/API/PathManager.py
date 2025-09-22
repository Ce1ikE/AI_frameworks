from pathlib import Path
import logging


class PathManager:
    logger = logging.getLogger(__name__)

    def __init__(self, entrypoint: str, config: str = "config.toml"):
        self.entrypoint_dir = Path(entrypoint).resolve().parent
        self.config = self.entrypoint_dir / config
        self.dataset = None
        self.input = None
        self.output = None
        self.logs = None
        self.models = None
    
    def resolve_dirs(self, config: dict):
        try:
            self.dataset = Path(self.entrypoint_dir / config["settings"]["dataset_dir"]).resolve()
            self.input   = Path(self.entrypoint_dir / config["settings"]["input_dir"]).resolve()
            self.output  = Path(self.entrypoint_dir / config["settings"]["output_dir"]).resolve()
            self.logs    = Path(self.entrypoint_dir / config["settings"]["log_dir"]).resolve()
            self.models  = Path(self.entrypoint_dir / config["settings"]["model_dir"]).resolve()

            for p in [self.dataset, self.input, self.output, self.logs, self.models]:
                p.mkdir(parents=True, exist_ok=True)
            self.logger.info("All required directories resolved and ensured.")
        except KeyError as e:
            raise ValueError(f"Missing configuration key: {e}")
  