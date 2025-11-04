from .base import FaceClassifier, LearningType
from ..utils.model_backend import BackendType

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

class KMeansClassifier(FaceClassifier):
    # mark as validated so FaceClassifier.__init__ won't raise
    def __init__(self, n_clusters=5, random_state=42):
        super().__init__(
            BackendType.SKLEARN, 
            LearningType.UNSUPERVISED
        )
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self._metrics = {}

    def train(self, embeddings, labels=None):
        self.model.fit(embeddings)

    def predict(self, embedding):
        cluster_id = self.model.predict([embedding])[0]
        return f"cluster_{cluster_id}"
