import logging
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from pathlib import Path
from ..API.FaceClassifier import FaceClassifier

class KMeansClassifier(FaceClassifier):
    logger = logging.getLogger(__name__)
    
    def __init__(self, n_clusters=5):
        super().__init__(model_name=__class__.__name__)
        self.model = None

    def predict(self, embeddings: list) -> int:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(embeddings)

    def train(self, embeddings: list, n_clusters: int = None, random_state: int = 42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.random_state = random_state
        self.logger.info(f"Initialized KMeans model with n_clusters={n_clusters}.")
        self.model = self.model.fit(embeddings)
        self.logger.info(f"Trained KMeans model with n_clusters={n_clusters}.")

    def evaluate(self, embeddings: list) -> dict:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        predictions = self.model.predict(embeddings)
        silhouette = silhouette_score(embeddings, predictions) if len(set(predictions)) > 1 else -1

        self.logger.info("Evaluated KMeans classifier.")
        return {
            "silhouette_score": silhouette,
            "cluster_centers": self.model.cluster_centers_,
            "cluster_labels": self.model.labels_
        }

    def settings(self):
        return {
            "n_clusters": len(self.model.cluster_centers_) if self.model else None,
            "max_iter": self.model.n_iter_ if self.model else None,
            "tol": self.model.tol if self.model else None,
            "random_state": self.random_state if self.random_state else None,
        }

