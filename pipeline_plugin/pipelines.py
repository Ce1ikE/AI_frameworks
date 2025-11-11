
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
                        device="cuda",
                        nms_threshold=0.4
                    )
                ),
                ImageFacesExtractor(
                    "Face extractor",
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
                        metric=Metric.EUCLIDEAN,
                        threshold=1.0
                    )
                )
            ],
            worker_sinks=[
                AnnotatedImageExporter("annotated image exporter"),
                ClassifiedAnnotatedImageExporter("annotated classified image"),
                worker_ex,
            ],
            pipeline_sinks=[
                WorkerAggregator("aggregator"),
                InferenceEvaluator("inference eval"),
                DimensionalityVisualizer("dimensionality reducer visualizer"),
                EmbeddingEvaluator(
                    "embedding evaluator",
                    neighbors=15,
                    max_k=20,
                    confidence_score_bins=50,
                    norm_distribution_bins=30
                ),
            ],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH,max_workers=2)


def streaming_pipeline(cluster_centers):
    from uniface import face_alignment
    import time

    prev_frame_time = 0
    new_frame_time = 0
    detector = RetinaFaceDetector(
        model_dir=Py4MLP.paths.models,
        model_name=RetinaFaceWeights.RESNET18,
        confidence_threshold=0.8,
        device="cuda",
        nms_threshold=0.5
    )    
    embedder = ArcFaceEmbedder(
        model_dir=Py4MLP.paths.models,
        model_name=ArcFaceWeights.RESNET50,
        device="cuda"
    )
    classifier = MetricClassifier(
        cluster_centers=cluster_centers,
        metric=Metric.EUCLIDEAN
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(0)

    path_to_video = Path("./recording.mp4")
    if path_to_video.is_file():
        path_to_video.unlink()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter.fourcc(c1='M',c2='P',c3='4',c4='V')
    isColor = True
    out = cv2.VideoWriter(path_to_video.as_posix(), fourcc, fps, (frame_width, frame_height),isColor)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        new_frame_time = time.time()
        bboxes, scores, landmarks = detector.detect_faces(
            ImageMessage(
                image=frame
            )
        )
    
        for bbox, score, landmark in zip(bboxes.astype(int),scores,landmarks.astype(int)):
            x1, y1, x2, y2 = bbox
            
            if landmark is not None and len(landmark) == 5:
                face, _ = face_alignment(frame, landmark)
            else:
                face = frame[y1:y2, x1:x2]

            embedding_face = embedder.embed_face(
                FaceMessage(
                    face_image=ImageMessage(
                        image=face
                    )
                )
            )

            prediction_label = classifier.predict(embedding_face)
            
            thickness = max(1, int(min(x2 - x1, y2 - y1) / 40))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            label = f"{score:.2f}"
            if label:
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            if landmark is not None:
                for (x, y) in landmark:
                    cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

            cv2.putText(
                frame,
                prediction_label,
                (x1, min(frame.shape[0], y2 + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()