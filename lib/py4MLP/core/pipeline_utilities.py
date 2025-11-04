import networkx as nx
from typing import Type
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import datetime as dt

@dataclass
class _SampleStorage:
    sample_dir: Path
    worker_storage: dict

class PipelineStorage:

    def __init__(self):
        self.sample_storage: defaultdict[int, _SampleStorage] = defaultdict(_SampleStorage)
        self.pipeline_ctx: defaultdict[str, list] = defaultdict(list)
        self.pipeline_path: Path = None
        self.pipeline_start_time: dt.datetime = None
        self.pipeline_end_time: dt.datetime = None
        self.pipeline_composition: dict = None

    def make_pipeline_storage(self,name: str,base_path: Path):
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pipeline_path = base_path / f"{name}_{timestamp}"
        self.pipeline_path.mkdir(parents=True, exist_ok=True)
        return self.pipeline_path

    def get_pipeline_storage(self):
        return self.pipeline_path

    def make_sample_storage(self,sample_id: int):
        sample_dir = self.pipeline_path / f"sample_{sample_id:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        self.sample_storage[sample_id] = _SampleStorage(
            sample_dir=sample_dir,
            worker_storage=defaultdict(dict)
        )
        return sample_dir

    def get_sample_storage(self,sample_id: int) -> _SampleStorage | None:
        return self.sample_storage.get(sample_id, None)
