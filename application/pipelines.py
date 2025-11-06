from lib.py4MLP.py4MLP import Py4MLP 
core = Py4MLP(entrypoint=__file__, enable_logging=False)

from lib.py4MLP.core.pipeline import *
from application import *
from application.detectors.retinaface import *
from application.embedders.arcface import *

def feature_extraction_pipeline(input_files_train):
    Pipeline(
        name=f"feature extraction pipeline",
        output_root=core.paths.output,
        source=ImageFileLoader("image loader",input_files_train),
        pipeline_spec=PipelineSpec(
            transfomers=[
                ImageFacesDetector(
                    "Face detector",
                    RetinaFaceDetector(
                        model_dir=core.paths.models,
                        model_name=RetinaFaceWeights.MNET_025,
                        confidence_threshold=0.7,
                        device="cpu"
                    )
                ),
                ImageFacesExtractor(
                    "Face extractor"
                ),
                ImageFacesEmbedder(
                    "Face embedder",
                    ArcFaceEmbedder(
                        model_dir=core.paths.models,
                        model_name=ArcFaceWeights.W600K_MBF,
                        device="cpu"
                    )
                )
            ],
            worker_sinks=[
                AnnotatedImageExporter("annotated image exporter"),
                CroppedFaceExporter("face image exporter"),
                EmbeddingExporter("embeddings exporter")
            ],
            pipeline_sinks=[
                DataAggregator("aggregator"),
                ParquetExporter("parquet aggregator exporter"),
                Reporter("reporter")
            ],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH)

def evaluation_pipeline(input_embeddings_files):
    Pipeline(
        name="embeddings evaluation pipeline",
        output_root=core.paths.output,
        source=EmbeddingFileLoader("embedding loader", input_embeddings_files),
        pipeline_spec=PipelineSpec(
            transfomers=[
                EmbeddingNormalizer("normalize embeddings"),
            ],
            worker_sinks=[
                EmbeddingExporter("normalized embeddings exporter"),
                EmbeddingEvaluator("embeddings evaluator"),
                TSNEVisualizer("t-SNE plot"),
                UMAPVisualizer("UMAP plot")
            ],
            pipeline_sinks=[],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH)

def training_pipeline(input_embeddings_files,n_clusters=13):
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
    RANDOM_STATE=42

    Pipeline(
        name="training pipeline",
        output_root=core.paths.output,
        source=EmbeddingFileLoader("embedding loader", input_embeddings_files),
        pipeline_spec=PipelineSpec(
            transfomers=[
                EmbeddingTrainer(
                    "model trainer",
                    reduce_to=-1,
                    algorithms=[
                        KMeans(n_clusters=n_clusters,random_state=RANDOM_STATE),
                        AgglomerativeClustering(n_clusters=n_clusters),
                        SpectralClustering(n_clusters=n_clusters,random_state=RANDOM_STATE),
                    ],
                )
            ],
            worker_sinks=[
                TSNEVisualizer("t-SNE plot"),
                # UMAPVisualizer("UMAP plot"),
                TrainingEvaluator("Training evaluator"),
                TrainingResultsExporter("results exporter")
            ],
            pipeline_sinks=[
                PipelineReporter("reporter")
            ],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH,max_workers=None)

def inference_pipeline(input_files_test,cluster_centers):
    Pipeline(
        name=f"inference pipeline",
        output_root=core.paths.output,
        source=ImageFileLoader("image loader",input_files_test),
        pipeline_spec=PipelineSpec(
            transfomers=[
                ImageFacesDetector(
                    "Face detector",
                    RetinaFaceDetector(
                        model_dir=core.paths.models,
                        model_name=RetinaFaceWeights.MNET_025,
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
                        model_dir=core.paths.models,
                        model_name=ArcFaceWeights.W600K_MBF,
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
                CroppedFaceExporter("face image exporter"),
                EmbeddingExporter("embeddings exporter"),
                ClassificationExporter("classification exporter")
            ],
            pipeline_sinks=[
                DataAggregator("aggregator"),
                ParquetExporter("parquet aggregator exporter"),
                Reporter("reporter")
            ],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH,max_workers=2)