import logging
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from pathlib import Path
from ..API.FaceClassifier import FaceClassifier
from ..API.Preprocessor import Preprocessor

# this classifier loads a pre-trained KMeans model from an ONNX file
# and uses it to classify face embeddings
class LoadedClassifier(FaceClassifier):
    logger = logging.getLogger(__name__)
    
    def __init__(self, model_path: Path):
        self.sess = Preprocessor.load_model(model_path.as_posix())
        super().__init__(model_name=model_path.stem)
        self.model = None

    def predict(self, embeddings: list) -> int:
        if self.sess is None:
            raise ValueError("Session is not initialized yet.")
        return self.sess.run(self.model, embeddings)

    def train(self, embeddings: list, n_clusters: int = None):
        raise NotImplementedError("This classifier does not support training.")

    def evaluate(self, embeddings: list) -> dict:
        if self.sess is None:
            raise ValueError("Session is not initialized yet.")
        predictions = self.sess.run(self.model, embeddings)
        silhouette = silhouette_score(embeddings, predictions) if len(set(predictions)) > 1 else -1

        self.logger.info("Evaluated KMeans classifier.")
        return {
            "silhouette_score": silhouette,
            "cluster_centers": "cluster_centers not available",
            "cluster_labels": "cluster_labels not available"
        }
    
    def settings(self):
        return {
            "model_path": self.sess if self.sess else None,
        }