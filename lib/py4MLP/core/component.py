from .pipeline_utilities import PipelineStorage

from typing import Any
from pathlib import Path
from collections import defaultdict

class BaseComponent:
    def __init__(self, name: str):
        self.name = name
        self.input_type = None
        self.output_type = None

    def process(self, data: Any) -> Any:
        raise NotImplementedError
    
    def settings(self) -> dict:
        return {}
    
class Source(BaseComponent):
    def __init__(self, name: str):
        super().__init__(name)

    def process(self):
        """Should yield or return data items one by one"""
        raise NotImplementedError

class Sink(BaseComponent):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name

    def process(self,*args,**kwargs):
        raise NotImplementedError
    
class PipelineSink(BaseComponent):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.pipeline_storage: PipelineStorage = None

    def process(self,data):
        """Consumes data (saves, prints, sends, etc.)"""
        raise NotImplementedError
    
class WorkerSink(BaseComponent):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.sample_id: Path
        self.sample_dir: Path
        self.pipeline_path: Path = None
        self.worker_storage: dict[str,list] = None
        self.worker_ctx = None

    def process(self,data):
        """Consumes data (saves, prints, sends, etc.)"""
        raise NotImplementedError
    
class Transformer(BaseComponent):
    def __init__(self, name):
        super().__init__(name)

    def process(self, data):
        """Transforms input data"""
        raise NotImplementedError