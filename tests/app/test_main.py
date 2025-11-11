import sys
import types
import importlib
from pathlib import Path

# To avoid importing heavy pipeline_plugin internals and their third-party deps
# insert a lightweight fake package for pipeline_plugin.pipelines so that
# ``from pipeline_plugin.pipelines import *`` in main.py resolves to our fakes.
fake_pkg = types.ModuleType('pipeline_plugin')
fake_pipelines = types.ModuleType('pipeline_plugin.pipelines')

def _fake_feature_extraction(input_files, output_path):
    return None

def _fake_training(input_embeddings, n_clusters, output_path):
    return None

def _fake_inference(input_files, cluster_centers, output_path):
    return None

def _fake_streaming(cluster_centers):
    return None

setattr(fake_pipelines, 'feature_extraction_pipeline', _fake_feature_extraction)
setattr(fake_pipelines, 'training_pipeline', _fake_training)
setattr(fake_pipelines, 'inference_pipeline', _fake_inference)
setattr(fake_pipelines, 'streaming_pipeline', _fake_streaming)

sys.modules['pipeline_plugin'] = fake_pkg
sys.modules['pipeline_plugin.pipelines'] = fake_pipelines

import main


def test_run_inference_calls_pipeline(monkeypatch):
    called = {}

    def fake_inference(input_files, clusters, output):
        # verify signatures look reasonable
        called['inference'] = (list(input_files), isinstance(clusters, dict), isinstance(output, Path))

    monkeypatch.setattr(main, 'inference_pipeline', fake_inference)
    main.run_inference()
    assert 'inference' in called


def test_run_training_and_feature_extraction_monkeypatched(monkeypatch):
    called = {}

    def fake_training(embeddings, n_clusters, output_path):
        called['training'] = (list(embeddings), n_clusters, isinstance(output_path, Path))

    def fake_feature_extraction(input_files, output_path):
        called['feature'] = (list(input_files), isinstance(output_path, Path))

    monkeypatch.setattr(main, 'training_pipeline', fake_training)
    monkeypatch.setattr(main, 'feature_extraction_pipeline', fake_feature_extraction)

    main.run_training()
    main.run_feature_extraction()

    assert 'training' in called
    assert 'feature' in called
