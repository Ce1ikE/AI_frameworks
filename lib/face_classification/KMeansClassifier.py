import logging
from xml.parsers.expat import model
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from pathlib import Path
from ..API.FaceClassifier import FaceClassifier
from ..API.Preprocessor import Preprocessor
from onnxruntime import InferenceSession
import numpy as np

# this classifier uses a pre-trained KMeans model:
# - either from an ONNX session (which is loaded from a file see Preprocessor.load_model)
# - or from a scikit-learn model
# and uses it to classify face embeddings
class KMeansClassifier(FaceClassifier):
    logger = logging.getLogger(__name__)
    cluster_centers = None
    inertia = None
    sess = None
    
    def __init__(self, model: KMeans | InferenceSession):
        super().__init__(model_name=__class__.__name__)
        if isinstance(model, KMeans):
            self.cluster_centers = model.cluster_centers_
            self.inertia = model.inertia_
            self.model = model
        if isinstance(model, InferenceSession):
            self.sess = model

    def predict(self, embedding: list) -> int:
        # either we use the ONNX runtime session or
        # the scikit-learn model to predict the label
        if embedding is None or len(embedding) == 0:
            raise ValueError("Embedding is None or empty")
        if self.sess is not None:
            return self.sess.run(self.model, embedding)
        elif self.model is not None:
            return self.model.predict(embedding)
        return -1

    def settings(self):
        return {
            "n_clusters": len(self.cluster_centers) if self.model else None,
            "max_iter": self.model.n_iter_ if self.model else None,
            "inertia": self.inertia if self.inertia else None,
        }
