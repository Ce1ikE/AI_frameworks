from datetime import datetime
from .bus import *
from .component import *
from .pipeline_utilities import *

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from enum import Enum
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

class _WorkerPipeline:
    """components to execute on the provided data"""
    def __init__(
        self,
        component_sequence: List[Transformer],
        sink_sequence: List[WorkerSink],
        pipeline_path: Path,  
    ):
        self.component_sequence = component_sequence
        self.sink_sequence = sink_sequence
        self.pipeline_path = pipeline_path
        self.databus = DataBus()

        for sub in sink_sequence:
            sub.pipeline_path = pipeline_path
            if sub.input_type is not None:
                for input_type in sub.input_type if isinstance(sub.input_type, list) else [sub.input_type]:
                    self.databus.subscribe(input_type, sub)
        
    def process(self, data, sample_id, sample_dir):
        worker_storage = defaultdict(list)
        self.databus.start()
        
        for sub in self.sink_sequence:
            sub.worker_storage = worker_storage
            sub.sample_dir = sample_dir
            sub.sample_id = sample_id

        try:
            current = data
            for component in self.component_sequence:
                current = component.process(current)
                self.databus.publish(current)

        except Exception as e:
            print(f"[WorkerPipelineProcess] Error processing sample {sample_id}: {e}")
            raise e

        self.databus.stop()
        return worker_storage , sample_id

_worker_instance = None 
def worker_init(
    component_sequence, 
    sink_sequence, 
    pipeline_path
):
    """
    Initializes the worker instance once per process.
    """
    global _worker_instance
    _worker_instance = _WorkerPipeline(
        component_sequence,
        sink_sequence,
        pipeline_path
    )

def _run_worker(
    data, 
    sample_id, 
    sample_dir,
):
    global _worker_instance
    if _worker_instance is None:
        raise RuntimeError("Worker not initialized")
    return _worker_instance.process(
        data, 
        sample_id, 
        sample_dir,
    )

@dataclass 
class PipelineSpec:
    transfomers: list[Transformer]
    worker_sinks: list[WorkerSink]
    pipeline_sinks: list[PipelineSink]


class PipelineEventType(Enum):
    PIPELINE_STARTED = "PIPELINE_STARTED"
    PIPELINE_FINISHED = "PIPELINE_FINISHED" 
    PIPELINE_FAILED = "PIPELINE_FAILED" 
    PIPELINE_BATCH_FINISHED = "PIPELINE_BATCH_FINISHED" 
    PIPELINE_SEQUENTIAL_FINISHED = "PIPELINE_SEQUENTIAL_FINISHED" 
    PIPELINE_STREAM_FINISHED = "PIPELINE_STREAM_FINISHED" 

class PipelineType(Enum):
    BATCH = "BATCH"
    STREAM = "STREAM"
    SEQUENTIAL = "SEQUENTIAL"

class Pipeline:
    def __init__(
        self,
        name: str,
        source: Source,
        output_root: Path,
        pipeline_spec: PipelineSpec 
    ):
        self.name = name.lower().replace(' ', '_')
        self.transfomers = pipeline_spec.transfomers
        self.source = source
        
        self.worker_sinks = pipeline_spec.worker_sinks
        self.pipeline_sinks = pipeline_spec.pipeline_sinks

        self.data_bus = DataBus()
        self.output_root = output_root

        self.pipeline_storage = PipelineStorage()
        self.all_storages = []
        self._is_built = False

    def print_info(self):
        print("Pipeline transformer sequence: \n" + "=" * 20)
        for i ,transformer in enumerate(self.transfomers):
            print(f"({i + 1}) Transformer : {transformer.__class__.__name__}")
        self.data_bus.print_info()
        input("(Press enter to continue)")

    def build_pipeline(self):
        """
        Validate and build the pipeline structure
        """
        if not self.transfomers:
            raise ValueError("""Pipeline must have at least one component to build""")
        if self.source is None:
            raise ValueError("""Pipeline must have a source component to build""")
        
        self.pipeline_storage.make_pipeline_storage(
            self.name,
            self.output_root
        )
        self.pipeline_storage.pipeline_composition = {
            "pipeline_name": self.name,
            "pipeline_components": [
                [t.__class__.__name__, t.name, t.settings() if t else "" ] 
                for t in self.transfomers
            ]
        }

        for sub in self.pipeline_sinks:
            if sub.input_type is not None:
                for input_type in sub.input_type if isinstance(sub.input_type, list) else [sub.input_type]:
                    self.data_bus.subscribe(
                        input_type,
                        sub
                    )
                sub.pipeline_storage = self.pipeline_storage

        self.print_info()
        
        self._is_built = True
        return self
    
    def sequential_pipeline(self):
        worker = _WorkerPipeline(
            self.transfomers,
            self.worker_sinks,
            self.pipeline_storage.pipeline_path
        )
        for i, data in enumerate(self.source.process()):
            sample_id = i + 1
            sample_dir = self.pipeline_storage.make_sample_storage(sample_id)
            worker_storage, _ = worker.process(
                data, 
                sample_id,
                sample_dir
            )
            self.pipeline_storage.get_sample_storage(sample_id).worker_storage = worker_storage
        self.data_bus.publish(PipelineEventType.PIPELINE_SEQUENTIAL_FINISHED)

    def stream_pipeline(self):
        raise NotImplementedError
        self.data_bus.publish(PipelineEventType.PIPELINE_STREAM_FINISHED)

    def batch_pipeline(self,max_workers):
        components = self.transfomers
        sinks = self.worker_sinks
        p_path = self.pipeline_storage.pipeline_path
        # https://superfastpython.com/processpoolexecutor-map-vs-submit/
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init, 
            initargs=(components, sinks, p_path)
        ) as executor:
            futures = []
            for i, data in enumerate(self.source.process()):
                sample_id = i + 1
                sample_dir = self.pipeline_storage.make_sample_storage(sample_id)
                futures.append(
                    executor.submit(
                        _run_worker, 
                        data, 
                        sample_id, 
                        sample_dir,
                    )
                )
            for f in as_completed(futures):
                worker_storage, sample_id = f.result()
                self.pipeline_storage.get_sample_storage(sample_id).worker_storage = worker_storage

        self.data_bus.publish(PipelineEventType.PIPELINE_BATCH_FINISHED)


    def run_pipeline(self,pipeline_type: PipelineType,max_workers=None):
        if not self._is_built:
            raise RuntimeError("""Pipeline must be built before running the pipeline""")

        self.data_bus.print_info()
        self.data_bus.start()

        self.data_bus.publish(PipelineEventType.PIPELINE_STARTED)
        self.pipeline_storage.pipeline_start_time = datetime.now()
        
        try:
            if pipeline_type == PipelineType.BATCH:
                self.batch_pipeline(max_workers)

            elif pipeline_type == PipelineType.SEQUENTIAL:
                self.sequential_pipeline()

            elif pipeline_type == PipelineType.STREAM:
                self.stream_pipeline()

            self.pipeline_storage.pipeline_end_time = datetime.now()
            self.data_bus.publish(PipelineEventType.PIPELINE_FINISHED)
        
        except Exception as e:
            self.data_bus.publish(PipelineEventType.PIPELINE_FAILED)
            print(f"[Pipeline] Error: {e}")
        
        finally:
            self.data_bus.stop()
            print("[Pipeline] Shutdown complete.")




        