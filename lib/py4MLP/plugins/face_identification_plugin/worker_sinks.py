from sklearn.metrics import silhouette_samples
from .dataclasses import *
from .transformers import *
from .utils.util_functions import *

import pandas as pd
import json
import plotly.express as px
from matplotlib import pyplot as plt
# https://www.geeksforgeeks.org/data-visualization/how-to-create-matplotlib-plots-without-a-gui/
import matplotlib
matplotlib.use('Agg')
plt.ioff()


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
class TSNEVisualizer(WorkerSink):
    def __init__(self, name: str, random_state: int = 42, perplexity: int = 30, n_iterations: int = 1000):
        super().__init__(name)
        self.input_type = NormalizedEmbeddings
        self.random_state = random_state
        self.perplexity = perplexity
        self.n_iterations = n_iterations
        self.fontdict = {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }

    def process(self, data: NormalizedEmbeddings):
        df: pd.DataFrame = data.embeddings
        embeddings = np.vstack(df["embedding"].values)
        
        if len(embeddings) <= self.perplexity:
            self.perplexity = max(5, len(embeddings) // 3)

        from sklearn.manifold import TSNE

        # ------- Plotting 2D ------- #
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
            s=5,
            alpha=0.8,
            marker='o',
            c='blue',
            edgecolors='none',
            linewidths=0.5
        )
        plt.title("t-SNE Visualization of Face Embeddings (2D)", fontdict=self.fontdict)
        plt.xlabel("t-SNE Dimension 1", fontdict=self.fontdict)
        plt.ylabel("t-SNE Dimension 2", fontdict=self.fontdict)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.sample_dir / "tsne_2d.svg", format="svg")
        plt.close()

        # -------------- plotting HTML 3D -------------- #
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
            title="t-SNE Visualization of Face Embeddings (3D)"
        )
        fig.update_traces(
            marker={
                'size': 2,
                'opacity': 0.8
            }
        )
        fig.write_html(self.sample_dir / "tsne_3d.html")

        # -------------- plotting 2D (with face images) -------------- #
        # if face images are provided, we'll use them as markers in custom plot
        # https://learnopencv.com/t-sne-for-feature-visualization/
        if "face_path" in df.columns:
            from PIL import Image

            face_paths = df["face_path"]
            # determine the range for x and y axes to properly place images
            x_min, x_max = projections_2d[:, 0].min(), projections_2d[:, 0].max()
            y_min, y_max = projections_2d[:, 1].min(), projections_2d[:, 1].max()
            x_span = x_max - x_min
            y_span = y_max - y_min
            thumb_frac = 0.05  # each thumbnail is 3% of the span
            thumb_w = x_span * thumb_frac
            thumb_h = y_span * thumb_frac
            fig = plt.figure(figsize=(10, 10))
            ax = plt.gca()
            ax.set_title("t-SNE Visualization with Face Thumbnails", fontdict=self.fontdict)
            for (x, y), img_path in zip(projections_2d, face_paths):
                img_path = Path(img_path)
                # if jpg doesn't exist, try jpeg
                if not img_path.exists():
                    img_path = img_path.with_suffix(".jpeg")

                try:
                    img = cv2.imread(str(img_path))
                except Exception as e:
                    print(f"Could not read image {img_path}: {e}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img.thumbnail((40, 40), Image.Resampling.LANCZOS)
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
            plt.savefig(self.sample_dir / "tsne_2d_faces.svg", format="svg")
            plt.close()

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class UMAPVisualizer(WorkerSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = NormalizedEmbeddings
        self.fontdict = {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }

    def process(self, data: NormalizedEmbeddings):
        import umap

        df: pd.DataFrame = data.embeddings
        embeddings = np.vstack(df["embedding"].values)

        # ------- Plotting 2D ------- #
        reducer = umap.UMAP(random_state=42,n_components=2)
        embedding_2d = reducer.fit_transform(embeddings)

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(
            embedding_2d[:, 0], 
            embedding_2d[:, 1],
            s=5,
            alpha=0.8,
            marker='o',
            c='blue',
            edgecolors='none',
            linewidths=0.5
        )
        plt.title("UMAP Visualization of Face Embeddings", fontdict=self.fontdict)
        plt.xlabel("UMAP Dimension 1", fontdict=self.fontdict)
        plt.ylabel("UMAP Dimension 2", fontdict=self.fontdict)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.sample_dir / "umap_2d.svg", format="svg")
        plt.close()

        # ------- Plotting 3D ------- #
        reducer_3d = umap.UMAP(random_state=42,n_components=3)
        embedding_3d = reducer_3d.fit_transform(embeddings)
        fig = px.scatter_3d(
            x=embedding_3d[:, 0],
            y=embedding_3d[:, 1],
            z=embedding_3d[:, 2],
            title="UMAP Visualization of Face Embeddings (3D)"
        )
        fig.update_traces(
            marker={
                "size": 2,
                "opacity": 0.8,
                "line": {"width": 0.5, "color": "darkgreen"}
            }
        )
        fig.write_html(self.sample_dir / "umap_3d.html")

        # ------- Plotting 2D with face images ------- #
        if "face_path" in df.columns:
            from PIL import Image

            face_paths = df["face_path"]
            # determine the range for x and y axes to properly place images
            x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
            y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
            x_span = x_max - x_min
            y_span = y_max - y_min
            thumb_frac = 0.05  # each thumbnail is 3% of the span
            thumb_w = x_span * thumb_frac
            thumb_h = y_span * thumb_frac
            fig = plt.figure(figsize=(10, 10))
            ax = plt.gca()
            ax.set_title("UMAP Visualization with Face Thumbnails", fontdict=self.fontdict)
            for (x, y), img_path in zip(embedding_2d, face_paths):
                img_path = Path(img_path)
                # if jpg doesn't exist, try jpeg
                if not img_path.exists():
                    img_path = img_path.with_suffix(".jpeg")

                try:
                    img = cv2.imread(str(img_path))
                except Exception as e:
                    print(f"Could not read image {img_path}: {e}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img.thumbnail((40, 40), Image.Resampling.LANCZOS)
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
            plt.savefig(self.sample_dir / "umap_2d_faces.svg", format="svg")
            plt.close()

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# class TrainingEvaluator(PipelineSink):
#     def __init__(self, name: str):
#         super().__init__(name)
#         self.input_type = PipelineEventType.PIPELINE_FINISHED

#     def process(self, event: PipelineEventType):
#         # Placeholder for future implementation
#         from sklearn.metrics import pairwise_distances
#         dists = pairwise_distances(embeddings, metric="cosine")
#         positive_dists = dists[labels[:, None] == labels[None, :]]
#         negative_dists = dists[labels[:, None] != labels[None, :]]

#         plt.figure(figsize=(10, 6))
#         sns.kdeplot(positive_dists, label="Same identity", shade=True)
#         sns.kdeplot(negative_dists, label="Different identities", shade=True)
#         plt.title("Embedding Distance Distributions")
#         plt.xlabel("Cosine Distance")
#         plt.ylabel("Density")
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         plt.savefig(out_path / "distance_distribution.svg", format="svg")
#         plt.close()


        # sample_silhouette_values = silhouette_samples(embeddings, labels)
        # y_lower = 10
        # n_clusters = len(np.unique(labels))
        # fig, ax = plt.subplots(figsize=(10, 6))
        # for i in range(n_clusters):
        #     ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        #     ith_cluster_silhouette_values.sort()
        #     size_cluster_i = ith_cluster_silhouette_values.shape[0]
        #     y_upper = y_lower + size_cluster_i
        #     color = plt.cm.nipy_spectral(float(i) / n_clusters)
        #     ax.fill_betweenx(
        #         np.arange(y_lower, y_upper),
        #         0,
        #         ith_cluster_silhouette_values,
        #         facecolor=color, edgecolor=color, alpha=0.7
        #     )
        #     ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        #     y_lower = y_upper + 10

        # ax.axvline(x=np.mean(sample_silhouette_values), color="red", linestyle="--")
        # ax.set_title("Silhouette Plot per Cluster")
        # ax.set_xlabel("Silhouette coefficient values")
        # ax.set_ylabel("Cluster label")
        # plt.savefig(out_path / "silhouette_analysis.svg", format="svg")
        # plt.close(fig)