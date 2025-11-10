from sklearn.metrics import silhouette_samples
from .dataclasses import *
from .transformers import *
from .utils.util_classes import *

import pandas as pd
import json
import plotly.express as px

# https://www.geeksforgeeks.org/data-visualization/how-to-create-matplotlib-plots-without-a-gui/
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.ioff()

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class WorkerAggregator(PipelineSink):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_type = [
            PipelineEventType.PIPELINE_BATCH_FINISHED, 
            PipelineEventType.PIPELINE_SEQUENTIAL_FINISHED,
        ]

    def process(self, event: PipelineEventType):
        all_dfs = []
        for sample_id, sample_info in self.pipeline_storage.sample_storage.items():
            worker_storage = sample_info.worker_storage
            parquet_files = worker_storage.get(WorkerKeys.EXTRACTION_RECORDS, [])
            for p in parquet_files:
                try:
                    df = pd.read_parquet(p)
                    df["sample_id"] = sample_id
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Could not read parquet {p}: {e}")

        if len(all_dfs) == 0:
            unified = pd.DataFrame()
        else:
            unified = pd.concat(all_dfs, ignore_index=True)
        
        output_path = self.pipeline_storage.pipeline_path / "processing_results.parquet"
        unified.to_parquet(output_path, index=False)
        print(f"[AggregatorReporter] Unified parquet saved: {output_path}")

        self.pipeline_storage.pipeline_ctx[PipelineKeys.AGGREGATED_RECORDS].append(output_path)

        time_diff = self.pipeline_storage.pipeline_end_time - self.pipeline_storage.pipeline_start_time
        report = {
            "pipeline": self.pipeline_storage.pipeline_composition,
            "total_images": len(self.pipeline_storage.sample_storage),
            "total_images_with_faces": len(unified["sample_id"].unique()) if not unified.empty else 0,
            "total_faces_detected": len(unified),
            "total_embeddings": len(unified),
            "embedding_dim": (
                unified[ExportKeys.EMBEDDING.value].iloc[0].shape[0] 
                if ExportKeys.EMBEDDING.value in unified.columns and len(unified) > 0 
                else None
            ),
            "start_time": str(self.pipeline_storage.pipeline_start_time),
            "end_time": str(self.pipeline_storage.pipeline_end_time),
            "duration_seconds": time_diff.total_seconds(),
        }

        # Save report
        report_path = self.pipeline_storage.pipeline_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4, cls=NpEncoder)

        print(f"[AggregatorReporter] Report saved at {report_path}")

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class EmbeddingEvaluator(PipelineSink):
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
        self.input_type = [
            PipelineEventType.PIPELINE_FINISHED
        ]
        self.max_k = max_k
        self.neighbors = neighbors
        self.fontdict = {
            "fontsize": 10,
            "fontweight": "bold",
            "fontfamily": "monospace",
        }

    def process(self, event: PipelineEventType.PIPELINE_FINISHED):
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans

        df: pd.DataFrame = pd.read_parquet(
            self.pipeline_storage.pipeline_ctx.get(PipelineKeys.AGGREGATED_RECORDS, [])[0]
        )

        save_dir = self.pipeline_storage.pipeline_path

        embeddings = np.vstack(df[ExportKeys.EMBEDDING.value].values)
        scores = df[ExportKeys.CONFIDENCE_SCORE.value].values
        embedding_norms = df[ExportKeys.EMBEDDING_NORM.value].values

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

        # Save evaluation plots and report
        # ---------- inertia plot ---------- #
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(2, self.max_k + 1), inertias, marker='o')
        plt.title("Elbow Method for Optimal k (Inertia)", fontdict=self.fontdict)
        plt.xlabel("Number of Clusters", fontdict=self.fontdict)
        plt.ylabel("Inertia", fontdict=self.fontdict)
        plt.xticks(range(2, self.max_k + 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_dir / "inertia_plot.svg", format="svg")
        plt.close(fig)

        # ---------- silhouette score plot ---------- #
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(2, self.max_k + 1), silhouette_scores, marker='o')
        plt.title("Silhouette Score Plot", fontdict=self.fontdict)
        plt.xlabel("Number of Clusters", fontdict=self.fontdict)
        plt.ylabel("Silhouette Score", fontdict=self.fontdict)
        plt.xticks(range(2, self.max_k + 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_dir / "silhouette_score_plot.svg", format="svg")
        plt.close(fig)

        # ---------- K-distance plot ---------- #
        fig = plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        plt.legend(
            [f"{n_neighbors}th Nearest Neighbor Distance" for n_neighbors in range(1, self.neighbors + 1)],
            bbox_to_anchor=(1.0, 0.5), loc='best'
        )
        plt.title("K-Distance Plot", fontdict=self.fontdict)
        plt.xlabel("Data Points sorted by Distance", fontdict=self.fontdict)
        plt.ylabel(f"Distance to Kth Nearest Neighbor", fontdict=self.fontdict)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(pad=2.0)
        plt.savefig(save_dir / f'k_distance_graph.svg', format='svg')
        plt.close(fig)

        # ---------- norm distribution plot ---------- #
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(range(len(embedding_norms)), embedding_norms, s=15, alpha=0.9)
        plt.title("Embedding Norm Distribution", fontdict=self.fontdict)
        plt.xlabel("Samples", fontdict=self.fontdict)
        plt.ylabel("Norm", fontdict=self.fontdict)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / "embedding_norm_scatter.svg", format="svg")
        plt.close(fig)

        # ---------- confidence score distribution plot ---------- #
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(range(len(scores)), scores, s=15, alpha=0.9)
        plt.title("Confidence Score Distribution", fontdict=self.fontdict)
        plt.xlabel("Sample Index", fontdict=self.fontdict)
        plt.ylabel("Confidence Score", fontdict=self.fontdict)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / "confidence_score_scatter.svg", format="svg")
        plt.close(fig)

        with open(save_dir / "evaluation_report.json", "w") as f:
            json.dump({
                "inertias": inertias,
                "silhouette_scores": silhouette_scores,
                "num_samples": len(embeddings),
                "mean_embedding_norm": float(np.mean(embedding_norms)),
                "mean_score": float(np.mean(scores)),
            }, f, indent=4)


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TrainingResultsAggregator(PipelineSink):
    def __init__(self, name):
        super().__init__(name)
        self.input_type = [
            PipelineEventType.PIPELINE_BATCH_FINISHED, 
            PipelineEventType.PIPELINE_SEQUENTIAL_FINISHED,
        ]

    def process(self, event: PipelineEventType):
        all_dfs = []
        for sample_id, sample_info in self.pipeline_storage.sample_storage.items():
            worker_storage = sample_info.worker_storage
            trained_embeddings = worker_storage.get(WorkerKeys.TRAINING_RECORDS, [])[0]
            self.pipeline_storage.pipeline_ctx[PipelineKeys.AGGREGATED_RECORDS].append(trained_embeddings)

        time_diff = self.pipeline_storage.pipeline_end_time - self.pipeline_storage.pipeline_start_time
        report = {
            "pipeline": self.pipeline_storage.pipeline_composition,
            "total_datasets": len(self.pipeline_storage.sample_storage),
            "start_time": str(self.pipeline_storage.pipeline_start_time),
            "end_time": str(self.pipeline_storage.pipeline_end_time),
            "duration_seconds": time_diff.total_seconds(),
        }

        # Save report
        report_path = self.pipeline_storage.pipeline_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4, cls=NpEncoder)

        print(f"[AggregatorReporter] Report saved at {report_path}")

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



