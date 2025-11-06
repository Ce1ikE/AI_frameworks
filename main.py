# main is the entrypoint of the application
# Core sets up the necessary components like the PathManager and logging 
# the Pipeline is where the actual work is done
from lib.py4MLP.py4MLP import Py4MLP 
core = Py4MLP(entrypoint=__file__, enable_logging=False)

from pathlib import Path

from lib.py4MLP.core.pipeline import *
from lib.py4MLP.plugins.face_identification_plugin import *
from lib.py4MLP.plugins.face_identification_plugin.detectors.retinaface import RetinaFaceWeights
from lib.py4MLP.plugins.face_identification_plugin.embedders.arcface import ArcFaceWeights

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

def main():
    # test_availability()
    
    # train_dir = Path("./dataset/train")
    # input_files_train = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.heic"))

    # feature_extraction_pipeline(input_files_train)
    retinaface_mnet_v1__arcface_w600k_mbf = Path(core.paths.output / "feature_extraction_pipeline_20251104_153054" / "retinaface_mnet_v1__arcface_w600k_mbf.parquet")
    retinaface_r34__arcface_w600k_r50 = Path(core.paths.output / "feature_extraction_pipeline_20251104_153904" / "retinaface_r34__arcface_w600k_r50.parquet")
    retinaface_mnet_v1__arcface_w600k_mbf = Path(core.paths.output / "feature_extraction_pipeline_20251104_154713" / "retinaface_mnet_v1__arcface_w600k_mbf.parquet")
    retinaface_mnet025__arcface_w600k_r50 = Path(core.paths.output / "feature_extraction_pipeline_20251104_163441" / "retinaface_mnet025__arcface_w600k_r50.parquet")
    retinaface_mnet050__arcface_w600k_r50 = Path(core.paths.output / "feature_extraction_pipeline_20251104_164225" / "retinaface_mnet050__arcface_w600k_r50.parquet")
    retinaface_r18__arcface_w600k_r50 = Path(core.paths.output / "feature_extraction_pipeline_20251104_165048" / "retinaface_r18__arcface_w600k_r50.parquet")
    retinaface_mnet_v2__arcface_w600k_r50 = Path(core.paths.output / "feature_extraction_pipeline_20251104_165601" / "retinaface_mnet_v2__arcface_w600k_r50.parquet")
    retinaface_r34__arcface_w600k_mbf = Path(core.paths.output / "feature_extraction_pipeline_20251104_170924" / "retinaface_r34__arcface_w600k_mbf.parquet")

    input_embeddings_files = [
        retinaface_mnet_v1__arcface_w600k_mbf,
        retinaface_r34__arcface_w600k_r50,
        retinaface_mnet_v1__arcface_w600k_mbf,
        retinaface_mnet025__arcface_w600k_r50,
        retinaface_mnet050__arcface_w600k_r50,
        retinaface_r18__arcface_w600k_r50,
        retinaface_mnet_v2__arcface_w600k_r50,
        retinaface_r34__arcface_w600k_mbf
    ]

    # evaluation_pipeline(input_embeddings_files)

    # training_pipeline(input_embeddings_files,13)

    test_dir = Path("./dataset/test")
    input_files_test = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.heic"))
    clusters = Path("C:\\Users\\ennis\\Documents\\Financien_Werk_OV\\my_works\\Courses TM\\2025-2026\\Semester 1\\AI frameworks\\AIFrameworks\\py4MLP_pipelines\\results\\training_pipeline_20251106_105917\\sample_001\\models\\cluster_center_data_20251106_110048.parquet")
    clusters: pd.DataFrame = pd.read_parquet(clusters)
    clusters = clusters[clusters.columns[0]].to_numpy()
    clusters = np.delete(clusters,4)
    print(len(clusters))
    # inference_pipeline(input_files_test,clusters)

    results_cls = Path("py4MLP_pipelines\\results\\inference_pipeline_20251106_171157\\classification_results.parquet")
    resulst_cls: pd.DataFrame = pd.read_parquet(results_cls)
    resulst_cls.info()
    print(resulst_cls.head())
    print(len(resulst_cls))

if __name__ == "__main__":
    main()

