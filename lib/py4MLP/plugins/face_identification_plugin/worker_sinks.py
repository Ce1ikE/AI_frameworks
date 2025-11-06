from .dataclasses import *
from .transformers import *
from .utils.util_functions import *

import pandas as pd
import json
import plotly.express as px
import datetime as dt
import seaborn as sns
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
# [ ] Save model settings	                	            
# [ ] Save trained model (ONNX)	            	            
# [ ] Save cosine similarity matrix	        
# [x] Compile results         	            
# [x] Save clustered embeddings	            
# [ ] Save classification labels (inference pipeline required)	        
# [ ] Save evaluation report (how good is the data ?)	                	            
# [ ] Save training report (what models used)	                	            
# [ ] Save inference report (what models used and time metrics)
# [ ] Save confusion matrix (requires labeled data though)	                
# [ ] Save ROC curve for different confidence thresholds (requires labeled data though)	                    
# [x] Save t-SNE visualization	            
# [x] Save UMAP visualization	            
# [x] Save silhouette scores	            
# [x] Save elbow method plot	            



# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Subscribers
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
            thickness = max(1, int(min(x2 - x1, y2 - y1) / 60))
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
        self.worker_storage[Keys.ANNOTATED_RECORDS].append(path_to_file)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CroppedFaceExporter(WorkerSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = ImageFaceMessage

    def process(self, data: ImageFaceMessage):
        dir = self.sample_dir / "cropped_faces"
        dir.mkdir(parents=True, exist_ok=True)
        for idx, face in enumerate(data.faces):
            path_to_file = dir / f"{data.original_image.path.stem}_face_{idx}.jpg"
            cv2.imwrite(str(path_to_file), face.face_image.image)
            self.worker_storage[Keys.FACE_RECORDS].append(path_to_file)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EmbeddingExporter(WorkerSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = [ImageEmbeddingMessage]

    def process(self, data: ImageEmbeddingMessage):
        dir = self.sample_dir / "embedding_faces"
        dir.mkdir(parents=True, exist_ok=True)
        for idx, e in enumerate(data.embeddings):
            embedding_path = dir / f"{data.original_image.path.stem}_face_{idx}.npy"
            np.save(embedding_path, e.embedding)
            self.worker_storage[Keys.EMBEDDINGS_RECORDS].append(embedding_path)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class NormalizedEmbeddingExporter(WorkerSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = [NormalizedEmbeddings]
    
    def process(self, data: NormalizedEmbeddings):
        dir = self.sample_dir / "normalized_embeddings"
        dir.mkdir(parents=True, exist_ok=True)
        data.embeddings.to_parquet(
            dir / "normalized_embeddings.parquet"
        )

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EmbeddingEvaluator(WorkerSink):
    """ 
    Evaluates normalized embeddings
    using K-distances, Silhouette scores, inertias, LOF, etc.
    """
    def __init__(
        self, 
        name: str, 
        max_k: int = 30,
        neighbors: int = 25
    ):
        super().__init__(name)
        self.input_type = NormalizedEmbeddings
        self.max_k = max_k
        self.neighbors = neighbors
        self.fontdict = {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }

    def process(self, data: NormalizedEmbeddings):
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans

        embeddings = np.vstack(data.embeddings["embedding"].values)

        # K-distance (for DBSCAN eps estimation)
        neigh = NearestNeighbors(n_neighbors=self.neighbors)
        model = neigh.fit(embeddings)
        distances, indices = model.kneighbors(embeddings)
        k_distances = np.sort(distances, axis=0)

        # Silhouette score and inertia for KMeans clustering
        inertias = []
        silhouette_scores = []
        for n_clusters in range(2, self.max_k + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        eval_data = ClusterEvaluationData(
            inertias=inertias,
            silhouette_scores=np.array(silhouette_scores),
            optimal_k_silhouette=int(np.argmax(silhouette_scores) + 2)  # +2 because range starts at 2
        )

        # Save evaluation plots and report
        # Inertia plot
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(2, self.max_k + 1), inertias, marker='o')
        plt.title("Elbow Method for Optimal k (Inertia)")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.xticks(range(2, self.max_k + 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.sample_dir / "inertia_plot.svg", format="svg")
        plt.close(fig)

        # Silhouette score plot
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(2, self.max_k + 1), silhouette_scores, marker='o')
        plt.title("Silhouette Score Plot")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.xticks(range(2, self.max_k + 1))
        plt.axvline(x=eval_data.optimal_k_silhouette, color='r', linestyle='--', label='Optimal k (Silhouette)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.sample_dir / "silhouette_score_plot.svg", format="svg")
        plt.close(fig)

        # K-distance plot
        fig = plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        legend = plt.legend(
            [f"{n_neighbors}th Nearest Neighbor Distance" for n_neighbors in range(1, self.neighbors + 1)],
            bbox_to_anchor=(1.0, 0.5), loc='upper left'
        )
        fig.add_artist(legend)
        plt.title("K-Distance Plot", fontdict=self.fontdict)
        plt.xlabel("Data Points sorted by Distance", fontdict=self.fontdict)
        plt.ylabel(f"Distance to Kth Nearest Neighbor", fontdict=self.fontdict)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(pad=2.0)
        plt.savefig(self.sample_dir / f'k_distance_graph.svg', format='svg')
        plt.close(fig)

        with open(self.sample_dir / "evaluation_report.json", "w") as f:
            json.dump({
                "inertias": eval_data.inertias,
                "silhouette_scores": eval_data.silhouette_scores.tolist(),
                "optimal_k_inertia": eval_data.optimal_k_inertia,
                "optimal_k_silhouette": eval_data.optimal_k_silhouette,
                "nr of data samples": len(embeddings),
                "pipeline" : data.source.stem
            }, f, indent=4)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TrainingResultsExporter(WorkerSink):
    def __init__(self, name):
        super().__init__(name)
        self.input_type = [TrainingResults]

    def process(self, data: TrainingResults):
        from skl2onnx import to_onnx

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.sample_dir / f"training_results_{timestamp}.json"
        model_dir = self.sample_dir / "models"
        model_dir.mkdir(parents=True,exist_ok=True)
        
        data.embeddings.embeddings.to_parquet(
            self.sample_dir / f"trained_embeddings{timestamp}.parquet"
        )

        results = {
            "timestamp": timestamp,
            "num_models": len(data.models),
            "models": [],
            "embedding_details": {},
        }

        X = np.vstack(data.embeddings.embeddings["embedding"].values).astype(np.float32)
        for trained_model in data.models:
            model_obj = getattr(trained_model, "model", None)
            model_name = getattr(trained_model, "model_name", "unknown_model")
            model_path = None
            cluster_centers = None

            model_path = model_dir / f"{model_name}_{timestamp}.onnx"
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


            cluster_center_data = {}
            if cluster_centers is not None:
                cluster_center_data[model_name] = cluster_centers.tolist()
                pd.DataFrame(cluster_center_data).to_parquet(
                   model_dir / f"cluster_center_data_{timestamp}.parquet"
                )
            
            model_info = {
                "model_name": model_name,
                "model_type": type(model_obj).__name__ if model_obj is not None else "Unknown",
                "training_time (seconds)": getattr(trained_model, "training_time", None),
                "model_path": str(model_path) if model_path else None,
                "has_cluster_centers": cluster_centers is not None,
            }
            results["models"].append(model_info)

        df = data.embeddings.embeddings
        embeddings = df["embedding"].values
        embedding_matrix = np.vstack(embeddings)

        results["embedding_details"] = {
            "num_embeddings": len(embeddings),
            "embedding_dim": embedding_matrix.shape[1] if len(embeddings) > 0 else None,
            "has_face_paths": "face_path" in df.columns,
            "source_file": data.embeddings.source.stem
        }

        with open(export_path, mode="w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        if Keys.TRAINING_RECORDS not in self.worker_storage:
            self.worker_storage[Keys.TRAINING_RECORDS] = []

        self.worker_storage[Keys.TRAINING_RECORDS].append({
            "timestamp": timestamp,
            "path": str(export_path),
            "summary": {
                "models": [m["model_name"] for m in results["models"]],
                "num_embeddings": results["embedding_details"]["num_embeddings"],
            }
        })

        print(f"[TrainingResultsExporter] Exported results to {export_path}")

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TSNEVisualizer(WorkerSink):
    def __init__(self, name: str, random_state: int = 42, perplexity: int = 30, n_iterations: int = 1000):
        super().__init__(name)
        self.input_type = [NormalizedEmbeddings,TrainingResults]
        self.random_state = random_state
        self.perplexity = perplexity
        self.n_iterations = n_iterations
        self.fontdict = {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }

    def plot_tsne_2d_visualisation(self,embeddings,label_colors,handles,suffix: str = ""):
        from sklearn.manifold import TSNE

        tsne_2d = TSNE(
            n_components=2,
            perplexity=self.perplexity,
            max_iter=self.n_iterations,
            random_state=self.random_state
        )
        projections_2d = tsne_2d.fit_transform(embeddings)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(
            projections_2d[:, 0], 
            projections_2d[:, 1],
            s=12,
            alpha=0.9,
            marker='o',
            c=label_colors,
            edgecolors='white',
            linewidths=0.2
        )
        if handles:
            plt.legend(handles=handles, title="Labels", loc="best", fontsize=8)
        plt.title(f"t-SNE Visualization of Face Embeddings (2D) {suffix}", fontdict=self.fontdict)
        plt.xlabel("t-SNE Dimension 1", fontdict=self.fontdict)
        plt.ylabel("t-SNE Dimension 2", fontdict=self.fontdict)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.sample_dir / f"tsne_2d_{suffix}.svg", format="svg")
        plt.close()

        return projections_2d

    def plot_tsne_3d_visualisation(self,embeddings,labels,suffix: str = ""):
        from sklearn.manifold import TSNE

        tsne_3d = TSNE(
            n_components=3,
            perplexity=self.perplexity,
            max_iter=self.n_iterations,
            random_state=self.random_state
        )
        projections_3d = tsne_3d.fit_transform(embeddings)

        fig = px.scatter_3d(
            x=projections_3d[:, 0],
            y=projections_3d[:, 1],
            z=projections_3d[:, 2],
            color=labels,
            title=f"t-SNE Visualization of Face Embeddings (3D) {suffix}"
        )
        fig.update_traces(
            marker={
                'size': 2,
                'opacity': 0.8
            }
        )
        fig.write_html(self.sample_dir / f"tsne_3d_{suffix}.html")

    def plot_tsne_2d_visualisation_with_thumbnails(self,projection_2d,df,label_colors,suffix: str = ""):
        # if face images are provided, we'll use them as markers in custom plot
        # https://learnopencv.com/t-sne-for-feature-visualization/
        from PIL import Image, ImageOps

        face_paths = df["face_path"]
        # determine the range for x and y axes to properly place images
        x_min, x_max = projection_2d[:, 0].min(), projection_2d[:, 0].max()
        y_min, y_max = projection_2d[:, 1].min(), projection_2d[:, 1].max()
        x_span = x_max - x_min
        y_span = y_max - y_min
        thumb_frac = 0.05  # each thumbnail is 3% of the span
        thumb_w = x_span * thumb_frac
        thumb_h = y_span * thumb_frac
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_title(f"t-SNE Visualization with Face Thumbnails {suffix}", fontdict=self.fontdict)
        for (x, y), img_path, color in zip(projection_2d, face_paths,label_colors):
            img_path = Path(img_path)
            # if jpg doesn't exist, try jpeg
            if not img_path.exists():
                img_path = img_path.with_suffix(".jpeg")

            try:
                img = cv2.imread(str(img_path))
            except Exception as e:
                print(f"Could not read image {img_path}: {e}")
                continue

            new_color = tuple([int(channel*255) for channel in color])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((40, 40), Image.Resampling.LANCZOS)
            img = img.convert("RGBA")
            img = ImageOps.expand(
                img, 
                border=5, 
                fill=new_color
            )
            ax.imshow(
                img,
                extent=(x - thumb_w/2, x + thumb_w/2, y - thumb_h/2, y + thumb_h/2),
                zorder=2,
                alpha=0.9
            )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(self.sample_dir / f"tsne_2d_faces_{suffix}.svg", format="svg")
        plt.close()

    def run(
        self,
        df: pd.DataFrame,
        embeddings,
        label_colors,
        handles,
        model_name: str = ""
    ):
        # ------- Plotting 2D ------- #
        projection_2d = self.plot_tsne_2d_visualisation(embeddings,label_colors,handles,model_name)
        # ------- Plotting 3D ------- #
        self.plot_tsne_3d_visualisation(embeddings,label_colors,model_name)
        # ------- Plotting 2D with face images ------- #
        if "face_path" in df.columns:
            self.plot_tsne_2d_visualisation_with_thumbnails(projection_2d,df,label_colors,model_name)

    def process(self, data: NormalizedEmbeddings | TrainingResults):
        if isinstance(data,TrainingResults):
            df: pd.DataFrame = data.embeddings.embeddings
            embeddings = np.vstack(df["embedding"].values)
            if len(embeddings) <= self.perplexity:
                raise ValueError("TSNE: perplexity is higer then the amount of samples")

            for trained_model in data.models:
                if trained_model.model_name in df.columns:
                    labels = df[trained_model.model_name]
                    labels = df[trained_model.model_name]
                    label_colors,handles = Utils.create_label_colors(labels)           
                    self.run(df,embeddings,label_colors,handles,trained_model.model_name)

        if isinstance(data,NormalizedEmbeddings):
            df: pd.DataFrame = data.embeddings
            embeddings = np.vstack(df["embedding"].values)
            if len(embeddings) <= self.perplexity:
                raise ValueError("TSNE: perplexity is higer then the amount of samples")
            self.run(df,embeddings,None,None,"")       

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class UMAPVisualizer(WorkerSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = [NormalizedEmbeddings,TrainingResults]
        self.fontdict = {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }

    def plot_umap_2d_visualisation(self,embeddings,label_colors,legend_handles, suffix: str = ""):
        import umap

        reducer = umap.UMAP(random_state=42,n_components=2)
        projection_2d = reducer.fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#111")
        plt.scatter(
            projection_2d[:, 0], 
            projection_2d[:, 1],
            s=12,
            alpha=0.9,
            marker='o',
            c=label_colors,
            edgecolors='white',
            linewidths=0.2
        )
        ax.set_facecolor("#111")
        ax.set_title(f"UMAP Visualization of Face Embeddings {suffix}", color="w", fontsize=13)
        ax.tick_params(colors="w", which="both")
        if legend_handles:
            plt.legend(handles=legend_handles, title="Labels", loc="best", fontsize=8)
        plt.title(f"UMAP Visualization of Face Embeddings {suffix}", fontdict=self.fontdict)
        plt.xlabel("UMAP Dimension 1", fontdict=self.fontdict)
        plt.ylabel("UMAP Dimension 2", fontdict=self.fontdict)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.sample_dir / f"umap_2d_{suffix}.svg", format="svg")
        plt.close()

        return projection_2d

    def plot_umap_3d_visualisation(self,embeddings,labels, suffix: str = ""):
        import umap

        reducer_3d = umap.UMAP(random_state=42,n_components=3)
        projection_3d = reducer_3d.fit_transform(embeddings)
        
        fig = px.scatter_3d(
            x=projection_3d[:, 0],
            y=projection_3d[:, 1],
            z=projection_3d[:, 2],
            color=labels,
            title=f"UMAP Visualization of Face Embeddings (3D) {suffix}"
        )
        fig.update_traces(
            marker={
                "size": 2,
                "opacity": 0.8,
                "line": {
                    "width": 0.5
                }
            }
        )
        fig.write_html(self.sample_dir / f"umap_3d_{suffix}.html")

    def plot_umap_2d_visualisation_with_thumbnails(self,projection_2d,df,label_colors,suffix: str = ""):
        from PIL import Image, ImageOps

        face_paths = df["face_path"]
        # determine the range for x and y axes to properly place images
        x_min, x_max = projection_2d[:, 0].min(), projection_2d[:, 0].max()
        y_min, y_max = projection_2d[:, 1].min(), projection_2d[:, 1].max()
        x_span = x_max - x_min
        y_span = y_max - y_min
        thumb_frac = 0.05  # each thumbnail is 3% of the span
        thumb_w = x_span * thumb_frac
        thumb_h = y_span * thumb_frac
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_title(f"UMAP Visualization with Face Thumbnails {suffix}", fontdict=self.fontdict)
        for (x, y), img_path, color in zip(projection_2d, face_paths,label_colors):
            img_path = Path(img_path)
            # if jpg doesn't exist, try jpeg
            if not img_path.exists():
                img_path = img_path.with_suffix(".jpeg")

            try:
                img = cv2.imread(str(img_path))
            except Exception as e:
                print(f"Could not read image {img_path}: {e}")
                continue
            new_color = tuple([int(channel*255) for channel in color])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((40, 40), Image.Resampling.LANCZOS)
            img = img.convert("RGBA")
            img = ImageOps.expand(
                img, 
                border=5, 
                fill=new_color
            )
            ax.imshow(
                img,
                extent=(x - thumb_w/2, x + thumb_w/2, y - thumb_h/2, y + thumb_h/2),
                zorder=2,
                alpha=0.9
            )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(self.sample_dir / f"umap_2d_faces_{suffix}.svg", format="svg")
        plt.close()


    def run(
        self,
        df: pd.DataFrame,
        embeddings,
        label_colors,
        handles,
        model_name = "",
    ):
        # ------- Plotting 2D ------- #
        projection_2d = self.plot_umap_2d_visualisation(embeddings,label_colors,handles,model_name)
        # ------- Plotting 3D ------- #
        self.plot_umap_3d_visualisation(embeddings,label_colors,model_name)
        # ------- Plotting 2D with face images ------- #
        if "face_path" in df.columns:
            self.plot_umap_2d_visualisation_with_thumbnails(projection_2d,df,label_colors,model_name)

    def process(self, data: NormalizedEmbeddings | TrainingResults):

        if isinstance(data,TrainingResults):
            df: pd.DataFrame = data.embeddings.embeddings
            embeddings = np.vstack(df["embedding"].values)

            for trained_model in data.models:
                if trained_model.model_name in df.columns:
                    labels = df[trained_model.model_name]
                    label_colors,handles = Utils.create_label_colors(labels)      
                    self.run(df,embeddings,label_colors,handles,trained_model.model_name)       

        if isinstance(data,NormalizedEmbeddings):
            df: pd.DataFrame = data.embeddings
            embeddings = np.vstack(df["embedding"].values)
            self.run(df,embeddings,None,None,"")       

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

    def process(self, data: TrainingResults):
        from sklearn.metrics import pairwise_distances, silhouette_samples
        df = data.embeddings.embeddings
        embeddings = np.vstack(df["embedding"].values)
        
        training_time_per_model = pd.DataFrame({
            "model" : [],
            "training time" : []
        })
        for trained_model in data.models:
            training_time_per_model.loc[len(training_time_per_model)] = [
                trained_model.model_name,
                trained_model.training_time
            ]
            
            if trained_model.model_name not in df.columns:
                continue
            labels = df[trained_model.model_name].to_numpy()
            if len(np.unique(labels)) <= 1:
                print(f"[Evaluator] {trained_model.model_name} contains only 1 label")
                continue
            print(f"[Evaluator] labels {len(np.unique(labels))}")

            sample_silhouette_values = silhouette_samples(embeddings, labels)
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
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i),fontdict=self.fontdict)
                y_lower = y_upper + 10

            ax.set_title(f"Silhouette Plot per Cluster {trained_model.model_name}",fontdict=self.fontdict)
            ax.set_xlabel("Silhouette coefficient values",fontdict=self.fontdict)
            ax.set_ylabel("Cluster label",fontdict=self.fontdict)
            plt.savefig(self.sample_dir / f"silhouette_analysis_{trained_model.model_name}.svg", format="svg")
            plt.close(fig)

        training_time_per_model.sort_values(by="training time", ascending=False,inplace=True)
        plt.figure(figsize=(10, 6))
        plt.title(f"Training time per model (in seconds)",fontdict=self.fontdict)
        plt.xlabel("Models")
        plt.yscale(value="log")
        plt.ylabel("Training time")
        plt.grid(visible=True, linestyle='--', alpha=0.7,axis="y")
        sns.barplot(
            data=training_time_per_model,
            x="model",
            y="training time",
            hue='training time',
            legend=False,
            edgecolor='black',
            linewidth=2,
        )
        plt.tight_layout()
        plt.savefig(self.sample_dir / f"training_time_per_model.svg", format="svg")
        plt.close()