from .dataclasses import *
from .transformers import *
from .utils.util_classes import *

import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
# https://www.geeksforgeeks.org/data-visualization/how-to-create-matplotlib-plots-without-a-gui/
import matplotlib
matplotlib.use('Agg')
plt.ioff()
plt.style.use("dark_background")


# TODO	                                
# ----
# [x] Save input images	                    
# [x] Save cropped faces	                
# [x] Save annotated image	                
# [x] Save embedding vectors	            
# [X] Save model settings	                	            
# [X] Save trained model (ONNX)	            	            
# [x] Compile results         	            
# [x] Save clustered embeddings	            
# [X] Save classification labels (inference pipeline required)	        
# [X] Save evaluation report (how good is the data ?)	                	            
# [X] Save training report (what models used)	                	            
# [ ] Save inference report (what models used and time metrics)
# [ ] Save ROC curve for different confidence thresholds (requires labeled data though)	                    
# [x] Save UMAP visualization	            
# [x] Save silhouette scores	            
# [x] Save elbow method plot	            
# [X] Add autolabel class (like in Deepbee to correct model's prediction) (input: dict of possibilities (name1,name2,etc...,others)) !!!

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class AnnotatedImageExporter(WorkerSink):
    def __init__(self, name):
        super().__init__(name)
        self.input_type = ImageDetectionMessage

    def process(self, data: ImageDetectionMessage):
        annotated_image = data.original_image.image.copy()
        for det in data.detections:
            x1, y1, x2, y2 = map(int, det.bbox.to_tuple())
            # calculate thickness proportional to the bbox size
            thickness = max(1, int(min(x2 - x1, y2 - y1) / 40))
            # draws bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            # draws confidence score of the detector (if available)
            label = f"{det.score:.2f}" if hasattr(det, "score") and det.score is not None else ""
            if label:
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            # draws the landmarks
            if det.landmarks is not None and len(det.landmarks) > 0:
                for (x, y) in det.landmarks:
                    cv2.circle(annotated_image, (int(x), int(y)), 3, (255, 0, 0), -1)

        path_to_file =  self.sample_dir / f"{data.original_image.path.stem}_annotated.jpg"
        cv2.imwrite(
            str(path_to_file), 
            annotated_image
        )
        self.worker_storage[WorkerKeys.ANNOTATED_RECORDS].append(path_to_file)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class WorkerExporter(WorkerSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = [
            ImageEmbeddingMessage,
            ImageClassifiedMessage
        ]

    def process(self, data: ImageEmbeddingMessage | ImageClassifiedMessage):
        image_name = data.original_image.path.stem
        rows = []
        # during feature extraction
        if isinstance(data,ImageEmbeddingMessage):
            for idx, msg in enumerate(data.embeddings):
                face_img = msg.face.face_image.image
                rows.append({
                    # same for each image
                    ExportKeys.IMAGE_NAME.value : image_name, 
                    # different per face image
                    ExportKeys.FACE_INDEX.value : idx, 
                    ExportKeys.EMBEDDING.value : msg.embedding, 
                    ExportKeys.EMBEDDING_NORM.value : float(np.linalg.norm(msg.embedding)), 
                    ExportKeys.BBOX.value : list(msg.face.detection.bbox.to_tuple()), 
                    ExportKeys.LANDMARKS.value : msg.face.detection.landmarks.tolist(), 
                    ExportKeys.CONFIDENCE_SCORE.value : float(msg.face.detection.score), 
                    ExportKeys.FACE_IMAGE.value : Utils.encode_img(face_img), 
                })    
        # during inference
        if isinstance(data,ImageClassifiedMessage):
            for idx, msg in enumerate(data.classifications):
                face_img = msg.embedding.face.face_image.image
                rows.append({
                    # same for each image
                    ExportKeys.IMAGE_NAME.value : image_name, 
                    # different per face image
                    ExportKeys.FACE_INDEX.value : idx, 
                    ExportKeys.EMBEDDING.value : msg.embedding.embedding, 
                    ExportKeys.EMBEDDING_NORM.value : float(np.linalg.norm(msg.embedding)), 
                    ExportKeys.BBOX.value : list(msg.embedding.face.detection.bbox.to_tuple()), 
                    ExportKeys.LANDMARKS.value : msg.embedding.face.detection.landmarks.tolist(), 
                    ExportKeys.CONFIDENCE_SCORE.value : float(msg.embedding.face.detection.score), 
                    ExportKeys.FACE_IMAGE.value : Utils.encode_img(face_img), 
                    ExportKeys.LABEL.value : msg.label, 
                })    

        if len(rows) != 0:
            path_to_file = self.sample_dir / f"{image_name}_results.parquet"
            pd.DataFrame(rows).to_parquet(path_to_file,index=False)     
            self.worker_storage.setdefault(WorkerKeys.EXTRACTION_RECORDS, []).append(path_to_file)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TrainingResultsExporter(WorkerSink):
    """
    We want to export the original dataframe but with labels
    save a model's centroids if possible and the model itself if supported
    """
    def __init__(self, name):
        super().__init__(name)
        self.input_type = [
            TrainingResults
        ]

    def process(self, data: TrainingResults):
        from skl2onnx import to_onnx

        # 1) export resulting dataframe
        path_to_file = self.sample_dir / f"trained_embeddings.parquet"
        data.embeddings.embeddings.to_parquet(path_to_file,index=False)
        self.worker_storage.setdefault(WorkerKeys.TRAINING_RECORDS, []).append(path_to_file)

        # 2) save models or centroids (cluster centers)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.sample_dir / "Models"
        model_dir.mkdir(parents=True,exist_ok=True)
        
        model_info = []
        X = np.vstack(data.embeddings.embeddings[ExportKeys.EMBEDDING.value].values)

        for trained_model in data.models:
            model_obj = trained_model.model
            model_name = trained_model.model_name
            model_path = model_dir / f"{model_name}.onnx"
            train_time = trained_model.training_time
            model_path = None
            cluster_centers = None

            try:
                onx = to_onnx(model_obj,X,target_opset=12)
                with open(model_path, "wb") as f:
                    f.write(onx.SerializeToString())
            except Exception as e:
                print(f"[Exporter] Warning: could not save model {model_name}: {e}")
                model_path = None

            if hasattr(model_obj, "cluster_centers_"):
                cluster_centers = model_obj.cluster_centers_
            elif hasattr(model_obj, "centroids_"):
                cluster_centers = model_obj.centroids_
            else:
                cluster_centers = None

            model_info.append({
                "model_name": model_name,
                "training_time__seconds": getattr(trained_model, "training_time", None),
                "cluster_centers": cluster_centers.tolist() if cluster_centers is not None else None,
            })
        
        pd.DataFrame(model_info).to_parquet(self.sample_dir / f"model_info.parquet")

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TrainingEvaluator(WorkerSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = TrainingResults
        self.fontdict = {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }

    def _plot_silhouette_analysis(self,embeddings,labels,model_name):
        from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
        sample_silhouette_values = silhouette_samples(embeddings, labels)
        silhouette_avg = silhouette_score(embeddings, labels)
        y_lower = 10
        n_clusters = len(np.unique(labels))
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.get_cmap("tab20")
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cmap(i % cmap.N)
            
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color, 
                edgecolor=color, 
                alpha=0.7,
                label=f"Cluster {i}"
            )
            ax.text(
                x=-0.05, 
                y=y_lower + 0.5 * size_cluster_i, 
                s=str(i),
                fontdict=self.fontdict
            )
            y_lower = y_upper + 10

        ax.set_title(f"Silhouette Plot per Cluster {model_name}",fontdict=self.fontdict)
        ax.set_xlabel("Silhouette coefficient values",fontdict=self.fontdict)
        ax.set_ylabel("Cluster label",fontdict=self.fontdict)
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_yticks([]) 
        ax.set_xticks([i*0.2 for i in range(-10,10 + 1)])
        ax.set_xlim((-1,1))
        ax.spines[['right', 'top']].set_visible(False)
        plt.savefig(self.sample_dir / model_name / f"silhouette_analysis_{model_name}.svg", format="svg")
        plt.close(fig)

    def _create_cluster_picture(self,df: pd.DataFrame,labels,model_name):
        cluster_ids = np.unique(labels)
        
        # normal image is 112x112
        for cluster_id in cluster_ids:
            cluster_df = df[labels == cluster_id]
            if len(cluster_df) == 0:
                continue
            
            # determine grid size
            n_images = len(cluster_df)
            cols = 8
            rows = int(np.ceil(n_images / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            axes = np.array(axes).reshape(rows, cols)

            fig.suptitle(f"{model_name} — Cluster {cluster_id}", fontsize=16)

            for ax in axes.flatten():
                ax.axis("off")

            for idx, encoded_img in enumerate(cluster_df[ExportKeys.FACE_IMAGE.value]):

                img = Utils.decode_img(encoded_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax = axes.flatten()[idx]
                ax.imshow(img)
                ax.axis("off")

            # Save PDF
            pdf_path = self.sample_dir / model_name / f"{model_name}_cluster_{cluster_id}.pdf"
            fig.savefig(pdf_path, format="pdf", bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {pdf_path}")

    def process(self, data: TrainingResults):
        df = data.embeddings.embeddings
        embeddings = np.vstack(df[ExportKeys.EMBEDDING.value].values)
        
        for trained_model in data.models:
            # here i do some primary checks whether i can actually plot 
            # and calculate silhouette values
            if trained_model.model_name not in df.columns:
                continue
            labels = df[trained_model.model_name].to_numpy()
            if len(np.unique(labels)) <= 1:
                print(f"[Evaluator] {trained_model.model_name} contains only 1 label")
                continue
            print(f"[Evaluator] labels {len(np.unique(labels))}")

            (self.sample_dir / trained_model.model_name).mkdir(parents=True,exist_ok=True)
            self._plot_silhouette_analysis(embeddings,labels,trained_model.model_name)
            self._create_cluster_picture(df,labels,trained_model.model_name)