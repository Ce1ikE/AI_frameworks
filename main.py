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
                        # AgglomerativeClustering(n_clusters=n_clusters),
                        # SpectralClustering(n_clusters=n_clusters),
                        # Birch(n_clusters=n_clusters),
                        # DBSCAN(eps=0.75,min_samples=15),
                    ],
                )
            ],
            worker_sinks=[
                TSNEVisualizer("t-SNE plot"),
                UMAPVisualizer("UMAP plot"),
                TrainingEvaluator("Training evaluator"),
                TrainingResultsExporter("results exporter")
            ],
            pipeline_sinks=[],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH)

def inference_pipeline():
    pass

def test_availability():
    import torch, onnxruntime, os, shutil, subprocess

    print("=== GPU Diagnostic ===")
    print("Torch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Torch device:", torch.cuda.get_device_name(0))
    else:
        print("Torch device: None")

    print("\nONNX Runtime providers:", onnxruntime.get_available_providers())
    # ONNX runtime allows us to use pytorch's (with cuda support) CUDA and cuDNN dll's 
    # because they are included with the package
    # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#compatibility-with-pytorch
    onnxruntime.preload_dlls()

    print("\nSystem PATH contains CUDA:", any("CUDA" in p for p in os.environ["PATH"].split(";")))

    nvcc = shutil.which("nvcc")
    print("nvcc found:", nvcc)
    if nvcc:
        subprocess.run(["nvcc", "--version"])  


def main():
    test_availability()
    
    # train_dir = Path("./dataset/train")
    # input_files_train = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.heic"))

    # feature_extraction_pipeline(input_files_train)

    embeddings_retinaface_mnet025_arcface_w600k_mbf_cleaned = Path(core.paths.output / "feature_extraction_pipeline_1" / "embeddings_retinaface_mnet025_arcface_w600k_mbf_cleaned.parquet")
    embeddings_retinaface_mnet050_arcface_w600k_mbf_cleaned = Path(core.paths.output / "feature_extraction_pipeline_2" / "embeddings_retinaface_mnet050_arcface_w600k_mbf_cleaned.parquet")
    input_embeddings_files = [
        embeddings_retinaface_mnet025_arcface_w600k_mbf_cleaned,
        embeddings_retinaface_mnet050_arcface_w600k_mbf_cleaned
    ]

    # evaluation_pipeline([embedding_retinaface_mnet050_arcface_w600k_mbf_cleaned])

    training_pipeline(input_embeddings_files,13)

    test_dir = Path("./dataset/test")
    input_files_test = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.heic"))

    # inference_pipeline()



if __name__ == "__main__":
    main()

