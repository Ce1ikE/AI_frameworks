import logging
from .base import FaceClassifier
import numpy as np
from enum import Enum
from sklearn.metrics.pairwise import (
    cosine_similarity, 
    euclidean_distances, 
    manhattan_distances, 
    paired_distances
)

class Metric(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT = "dot"

class MetricClassifier(FaceClassifier):
    def __init__(
        self, 
        cluster_centers: dict[str,np.ndarray], 
        metric: Metric = Metric.EUCLIDEAN, 
        threshold: float = None,

    ):
        super().__init__()
        self.metric = metric
        self.threshold = threshold

        if cluster_centers is None and len(cluster_centers) > 1:
            raise ValueError("Cluster centers cannot be None and there must be more then 1 cluster")

        if threshold is not None:
            if metric == Metric.EUCLIDEAN and threshold < 0:
                raise ValueError("Euclidean threshold must be >= 0")
            if metric == Metric.COSINE and not (-1 <= threshold <= 1):
                raise ValueError("Cosine threshold must be in [-1, 1]")


        self.cluster_centers: dict[str,np.ndarray] = {}
        for label, center in cluster_centers.items():
            center = np.asarray(center).flatten()
            if center.ndim != 1:
                raise ValueError(f"Center for '{label}' must be 1D, got shape {center.shape}")
            self.cluster_centers[label] = center

    def predict(self, embedding: np.ndarray) -> int:
        if embedding is None:
            return "Unknown"
        embedding = np.array(embedding).flatten()

        # first check every cluster center and compute the distance based on the metric provided
        # to the embedding then return the index of the closest center as the label
        # quick note on distance metrics:
        # - euclidean distance: range is [0, inf) where 0 means exactly the same, inf means exactly opposite
        # - cosine similarity: range is [-1, 1] where 1 means exactly the same, -1 means exactly opposite
        # - dot product: range is [-inf, inf) where inf means exactly the same, -inf means exactly opposite

        # there is also some other considerations we have to talk about explained in these posts on stackexchange:
        # https://stats.stackexchange.com/questions/232500/how-do-i-know-my-k-means-clustering-algorithm-is-suffering-from-the-curse-of-dim
        # https://stats.stackexchange.com/questions/99171/why-is-euclidean-distance-not-a-good-metric-in-high-dimensions
        # https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf

        # the main problem is that in higher dimensions the distance between 2 points becomes less meaningful
        # this is because the points become more sparse and the distance becomes greater between points
        # which means that the difference between the closest and farthest point becomes smaller
        # this makes it harder to distinguish between points and this certain metrics become less useful
        # for example in high dimensions the euclidean distance and cosine similarity becomes less useful because of the distance
        # the dot product takes into account both the magnitude and the direction of the vectors however makes it computationally more expensive
        results: dict[str, float] = {}
        if self.metric and self.cluster_centers is not None and embedding is not None:
            cluster_distances = []
            for label, center in self.cluster_centers.items():
                if self.metric == Metric.EUCLIDEAN:
                    score = -float(np.linalg.norm(embedding - center))
                
                elif self.metric == Metric.COSINE:
                    score = np.dot(embedding, center)

                elif self.metric == Metric.DOT:
                    score = float(np.dot(embedding, center))
                
                results[label] = score

            best_label = max(results, key=results.get)
            best_score = results[best_label]

            # so once we have all distances we can find the closest center
            # but we also need to check if the distance is below a certain threshold
            # because if the distance is too high, we are likely dealing with an unknown face
            print(f"best score: {best_score}")
            if self.threshold is not None:
                if self.metric == Metric.EUCLIDEAN:
                    # For Euclidean threshold is applied to raw distance
                    if -best_score > self.threshold:
                        return "none"
                else:
                    # Cosine/Dot -> higher is better
                    if best_score < self.threshold:
                        return "none"
            return best_label
    
    def settings(self):
        return {
            "n_clusters": len(self.cluster_centers) if self.cluster_centers is not None else None,
            "cluster_centers" : self.cluster_centers,
            "metric": self.metric.value if self.metric else None,
            "threshold": self.threshold,
        }
