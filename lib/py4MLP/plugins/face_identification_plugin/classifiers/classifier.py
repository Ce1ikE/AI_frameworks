import logging
from .base import FaceClassifier
import numpy as np
from enum import Enum
from ..utils.model_backend import BackendType
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
        cluster_centers: np.ndarray, 
        metric: Metric = Metric.EUCLIDEAN, 
        threshold: float = None,

    ):
        super().__init__(
            backend=BackendType.SKLEARN, 
        )
        self.metric = metric

        if cluster_centers is None and len(cluster_centers) > 0:
            raise ValueError("Cluster centers cannot be None")
        if threshold is not None and threshold < 0:
            raise ValueError("Threshold must be non-negative")
        if metric == Metric.COSINE and threshold is not None and (threshold < -1 or threshold > 1):
            raise ValueError("Cosine similarity threshold must be in the range [-1, 1]")
        
        self.cluster_centers = cluster_centers
        self.threshold = threshold

    def predict(self, embedding: np.ndarray) -> int:
        embedding = np.array(embedding).flatten().reshape(1, -1) if embedding is not None else None
        # classify using cluster centers and metric
        # each center is a numpy array of shape (d,)
        # where d is the dimension which means the requires the same dimensions
        if self.cluster_centers is not None and embedding is not None:
            if embedding.shape[1] != self.cluster_centers.shape[1]:
                raise ValueError(f"{self.get_name()} : Embedding dimension {embedding.shape[1]} does not match cluster centers dimension {self.cluster_centers.shape[1]}")

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

        if self.metric and self.cluster_centers is not None and embedding is not None:
            cluster_distances = []
            for center in self.cluster_centers:
                center = center.reshape(1, -1)
                if self.metric == Metric.EUCLIDEAN:
                    # if we flip the sign then -1 * [0, inf] becomes [-inf, 0] which results in the higher the value the more similar
                    # and thus we can use argmax to find the closest center for all metrics
                    dist = -1 * euclidean_distances(embedding, center)
                elif self.metric == Metric.COSINE:
                    dist = cosine_similarity(embedding, center)
                elif self.metric == Metric.DOT:
                    dist = np.dot(embedding, center)
                cluster_distances.append(dist)
            # so once we have all distances we can find the closest center
            # but we also need to check if the distance is below a certain threshold
            # because if the distance is too high, we are likely dealing with an unknown face
            if self.threshold is not None:
                min_distance = np.min(cluster_distances)
                if min_distance > self.threshold:
                    # unknown face
                    return -1
            return np.argmax(cluster_distances)
        # unknown face
        return -1  
    
    def settings(self):
        return {
            "n_clusters": len(self.cluster_centers) if self.cluster_centers is not None else None,
            "cluster_centers" : self.cluster_centers,
            "metric": self.metric.value if self.metric else None,
            "threshold": self.threshold,
        }
