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
        super().__init__(model_name=model_path.stem)
        self.sess = Preprocessor.load_model(model_path.as_posix())
        self.model = None

    def predict(self, embeddings: list) -> int:
        if self.sess is None:
            raise ValueError("Session is not initialized yet.")
        return self.sess.run(self.model, embeddings)

    def train(self, embeddings: list, n_clusters: int = None):
        raise NotImplementedError("This classifier does not support training.")

    def settings(self):
        return {
            "model_path": self.sess if self.sess else None,
        }