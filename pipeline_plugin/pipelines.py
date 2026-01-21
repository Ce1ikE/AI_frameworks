import time
from uniface import face_alignment

from lib.py4MLP.core.pipeline import *
from lib.py4MLP.py4MLP import Py4MLP 

from . import *
from pipeline_plugin.detectors.retinaface import *
from pipeline_plugin.detectors.violajones import *
from pipeline_plugin.detectors.yunet import *
from pipeline_plugin.embedders.arcface import *
from pipeline_plugin.depth_estimators.midas import *

RANDOM_STATE=42


def feature_extraction_pipeline(
        input_files_train,
        output_path,
        pipeline_name="feature extraction pipeline",
    ):

    Pipeline(
        name=pipeline_name,
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
                ImageDepthEstimator(
                    "Depth estimator",
                    MiDaSEstimator(
                        model_dir=Py4MLP.paths.models,
                        model_name=MiDaSWeights.DPT_SWIN2_TINY_256,
                        device="cuda",
                        depth_threshold=0.4
                    )
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
                DimensionalityVisualizer("dimensionality reducer visualizer UMAP", method=DimensionalityVisualizer.ReductionTechnique.UMAP),
                DimensionalityVisualizer("dimensionality reducer visualizer T-SNE", method=DimensionalityVisualizer.ReductionTechnique.TSNE),
            ],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH,max_workers=2)

def training_pipeline(
        input_embeddings_files: list[Path],
        n_clusters,
        output_path,
        pipeline_name="training pipeline",
    ):
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
        name=pipeline_name,
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
                DimensionalityVisualizer("dimensionality reducer visualizer UMAP", method=DimensionalityVisualizer.ReductionTechnique.UMAP),
                DimensionalityVisualizer("dimensionality reducer visualizer T-SNE", method=DimensionalityVisualizer.ReductionTechnique.TSNE),
            ],
            pipeline_sinks=[
                TrainingResultsAggregator("Training aggregator"),
            ],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH,max_workers=None)

def inference_pipeline(
        input_files_test,
        cluster_centers: dict,
        output_path,
        pipeline_name="inference pipeline",
    ):
    worker_ex = WorkerExporter("worker results exporter")
    worker_ex.input_type = [ImageClassifiedMessage]

    Pipeline(
        name=pipeline_name,
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
                        threshold=1.145
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
                InferenceEvaluator(
                    "inference eval",
                    cluster_centers=cluster_centers
                ),
                DimensionalityVisualizer("dimensionality reducer visualizer"),
                EmbeddingEvaluator(
                    "embedding evaluator",
                    neighbors=15,
                    max_k=20,
                    confidence_score_bins=50,
                    norm_distribution_bins=30,
                ),
            ],
        )
    ).build_pipeline().run_pipeline(PipelineType.BATCH,max_workers=2)


def streaming_pipeline(cluster_centers):
    """Real-time face detection, embedding, and classification pipeline using webcam."""

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

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter.fourcc('M', 'P', '4', 'V')
    out = cv2.VideoWriter(
        path_to_video.as_posix(),
        fourcc,
        fps,
        (frame_width, frame_height),
        True
    )

    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break

        new_frame_time = time.time()
        current_fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        bboxes, scores, landmarks = detector.detect_faces(ImageMessage(image=frame))

        for bbox, score, landmark in zip(
            bboxes.astype(int),
            scores,
            landmarks.astype(int)
        ):
            x1, y1, x2, y2 = bbox
            
            if landmark is not None and len(landmark) == 5:
                face, _ = face_alignment(frame, landmark)
            else:
                face = frame[y1:y2, x1:x2]

            embedding_face = embedder.embed_face(
                FaceMessage(face_image=ImageMessage(image=face))
            )
            prediction_label = classifier.predict(embedding_face)

            box_color = (0, 255, 100)
            box_thickness = max(2, int(min(x2 - x1, y2 - y1) / 60))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
            
            corner_length = min(20, (x2 - x1) // 4)
            accent_color = (100, 255, 255) 
            accent_thickness = box_thickness + 1
            
            cv2.line(frame, (x1, y1), (x1 + corner_length, y1), accent_color, accent_thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_length), accent_color, accent_thickness)
            
            cv2.line(frame, (x2, y1), (x2 - corner_length, y1), accent_color, accent_thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_length), accent_color, accent_thickness)
            
            cv2.line(frame, (x1, y2), (x1 + corner_length, y2), accent_color, accent_thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_length), accent_color, accent_thickness)
            
            cv2.line(frame, (x2, y2), (x2 - corner_length, y2), accent_color, accent_thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_length), accent_color, accent_thickness)

            if landmark is not None:
                for (x, y) in landmark:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 100, 100), -1)

            confidence_text = f"Conf: {score:.2%}"
            identity_text = f"ID: {prediction_label}"
            
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            font_thickness = 1
            padding = 5

            (conf_w, conf_h), conf_baseline = cv2.getTextSize(
                confidence_text, font, font_scale, font_thickness
            )
            label_y_top = max(conf_h + padding * 2, y1 - 5)
            
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x1, label_y_top - conf_h - padding * 2),
                (x1 + conf_w + padding * 2, label_y_top),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(
                frame,
                confidence_text,
                (x1 + padding, label_y_top - padding),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

            (id_w, id_h), id_baseline = cv2.getTextSize(
                identity_text, font, font_scale, font_thickness
            )
            label_y_bottom = min(frame.shape[0] - padding, y2 + id_h + padding * 2 + 5)
            
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x1, label_y_bottom - id_h - padding * 2),
                (x1 + id_w + padding * 2, label_y_bottom),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(
                frame,
                identity_text,
                (x1 + padding, label_y_bottom - padding),
                font,
                font_scale,
                accent_color,
                font_thickness,
                cv2.LINE_AA
            )

        fps_text = f"FPS: {int(current_fps)}"
        (fps_w, fps_h), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2
        )
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (fps_w + 15, fps_h + 15), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(
            frame,
            fps_text,
            (10, fps_h + 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (100, 255, 0),
            2,
            cv2.LINE_AA
        )

        out.write(frame)
        cv2.imshow('Face Recognition - Press Q to quit', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()