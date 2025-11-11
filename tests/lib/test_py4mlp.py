import sys
import types
import pytest
from pathlib import Path
from dataclasses import dataclass

# Provide a lightweight dummy for networkx which is imported at module import time
sys.modules.setdefault('networkx', types.ModuleType('networkx'))
_tqdm_mod = sys.modules.setdefault('tqdm', types.ModuleType('tqdm'))
setattr(_tqdm_mod, 'tqdm', lambda x: x)

from lib.py4MLP.py4MLP import Py4MLP, PathsConfig
from lib.py4MLP.core.pipeline import (
    Pipeline,
    PipelineSpec,
    PipelineType,
)
from lib.py4MLP.core.component import Source, Transformer, WorkerSink


def test_setup_paths_creates_expected_dirs(tmp_path):
    # ensure a clean entrypoint and paths config
    Py4MLP.entrypoint = str(tmp_path / "project_dir")
    Py4MLP.paths = PathsConfig()

    # run setup_paths and assert directories are created
    Py4MLP.setup_paths()

    assert Py4MLP.pipeline_library_dir.exists()
    assert Py4MLP.paths.output.exists()
    assert Py4MLP.paths.logs.exists()
    assert Py4MLP.paths.models.exists()


def test_pipeline_sequential_flow_and_build_errors(tmp_path, monkeypatch):
    # Avoid interactive prompt in Pipeline.print_info during tests
    monkeypatch.setattr(Pipeline, "print_info", lambda self: None)

    @dataclass
    class MyMessage:
        value: int

    class DummySource(Source):
        def __init__(self, items):
            super().__init__("dummy source")
            self._items = items

        def process(self):
            for it in self._items:
                yield it

    class DummyTransformer(Transformer):
        def __init__(self):
            super().__init__("dummy transformer")

        def process(self, data):
            # Return a dataclass so DataBus will route to subscribed sinks
            return MyMessage(value=data * 2)

    class DummyWorkerSink(WorkerSink):
        def __init__(self):
            super().__init__("dummy worker")
            # subscribe to MyMessage
            self.input_type = MyMessage

        def process(self, data):
            # collect values into the worker_storage dict provided by the pipeline
            if self.worker_storage is None:
                return
            self.worker_storage.setdefault("collected", []).append(data.value)

    # 1) build should raise when no transformers
    src = DummySource([1])
    spec_empty = PipelineSpec(transfomers=[], worker_sinks=[], pipeline_sinks=[])
    p_empty = Pipeline("p-empty", source=src, output_root=tmp_path, pipeline_spec=spec_empty)
    with pytest.raises(ValueError):
        p_empty.build_pipeline()

    # 2) full sequential flow
    src = DummySource([1, 2])
    spec = PipelineSpec(transfomers=[DummyTransformer()], worker_sinks=[DummyWorkerSink()], pipeline_sinks=[])
    pipeline = Pipeline("test-pipeline", source=src, output_root=tmp_path, pipeline_spec=spec)
    pipeline.build_pipeline()

    # run sequential pipeline and validate storage and worker outputs
    pipeline.run_pipeline(PipelineType.SEQUENTIAL)

    assert pipeline.pipeline_storage.pipeline_path is not None
    # sample storage for first sample should exist and contain worker results
    s1 = pipeline.pipeline_storage.get_sample_storage(1)
    assert s1 is not None
    assert isinstance(s1.worker_storage, dict)
    assert "collected" in s1.worker_storage
    # first sample: input 1 -> transformer produces 2
    assert 2 in s1.worker_storage["collected"]
