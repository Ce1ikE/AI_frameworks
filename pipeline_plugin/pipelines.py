
from lib.py4MLP.core.pipeline import *
from lib.py4MLP.py4MLP import Py4MLP 

from . import *
from pipeline_plugin.detectors.retinaface import *
from pipeline_plugin.embedders.arcface import *

RANDOM_STATE=42


def feature_extraction_pipeline(input_files_train,output_path):
    Pipeline(
        name=f"feature extraction pipeline",
        output_root=output_path,
        source=ImageFileLoader(
            "image loader",
            input_files_train
        ),
        pipeline_spec=PipelineSpec(
            transfomers=[
                ImageFacesDetector(
                    "Face detector",
                    RetinaFaceDetector(
                        model_dir=Py4MLP.paths.models,
                        model_name=RetinaFaceWeights.RESNET34,
                        confidence_threshold=0.7,
                        device="cuda"
                    )
                ),
                ImageFacesExtractor(
                    "Face extractor"
                ),
                ImageFacesEmbedder(
                    "Face embedder",
                    ArcFaceEmbedder(
                        model_dir=Py4MLP.paths.models,
                        model_name=ArcFaceWeights.RESNET50,
                        device="cuda"
                    )
                )
            ],
            worker_sinks=[
                AnnotatedImageExporter("annotated image exporter"),
                WorkerExporter("worker results exporter")
            ],
            pipeline_sinks=[
                WorkerAggregator("worker results aggregator"),
                EmbeddingEvaluator("embedding evaluator",neighbors=15,max_k=20),
                DimensionalityVisualizer("dimensionality reducer visualizer")
            ],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH,max_workers=1)

def training_pipeline(input_embeddings_files,n_clusters,output_path):
    from sklearn.cluster import (
        KMeans,
        DBSCAN,
        AgglomerativeClustering,
        MeanShift,
        OPTICS,
        SpectralClustering,
        Birch
    )
    from sklearn.cluster._hdbscan.hdbscan import HDBSCAN        

    Pipeline(
        name="training pipeline",
        output_root=output_path,
        source=EmbeddingFileLoader("embedding loader", input_embeddings_files),
        pipeline_spec=PipelineSpec(
            transfomers=[
                EmbeddingTrainer(
                    "model trainer",
                    algorithms=[
                        KMeans(n_clusters=n_clusters,random_state=RANDOM_STATE),
                        AgglomerativeClustering(n_clusters=n_clusters),
                    ],
                )
            ],
            worker_sinks=[
                TrainingResultsExporter("results exporter"),
                TrainingEvaluator("Training evaluator"),
                DimensionalityVisualizer("dimensionality reducer visualizer"),
            ],
            pipeline_sinks=[
                TrainingResultsAggregator("Training aggregator"),
            ],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH,max_workers=None)

def inference_pipeline(input_files_test,cluster_centers: dict,output_path):
    worker_ex = WorkerExporter("worker results exporter")
    worker_ex.input_type = [ImageClassifiedMessage]
    
    Pipeline(
        name=f"inference pipeline",
        output_root=output_path,
        source=ImageFileLoader("image loader",input_files_test),
        pipeline_spec=PipelineSpec(
            transfomers=[
                ImageFacesDetector(
                    "Face detector",
                    RetinaFaceDetector(
                        model_dir=Py4MLP.paths.models,
                        model_name=RetinaFaceWeights.RESNET34,
                        confidence_threshold=0.7,
                        device="cuda"
                    )
                ),
                ImageFacesExtractor(
                    "Face extractor"
                ),
                ImageFacesEmbedder(
                    "Face embedder",
                    ArcFaceEmbedder(
                        model_dir=Py4MLP.paths.models,
                        model_name=ArcFaceWeights.RESNET50,
                        device="cuda"
                    )
                ),
                EmbeddingClassifier(
                    "Face classifier",
                    MetricClassifier(
                        cluster_centers=cluster_centers,
                        metric=Metric.EUCLIDEAN
                    )
                )
            ],
            worker_sinks=[
                AnnotatedImageExporter("annotated image exporter"),
                worker_ex,
            ],
            pipeline_sinks=[
                WorkerAggregator("aggregator"),
                InferenceEvaluator("inference eval"),
            ],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH,max_workers=2)