from .ModelWrapper import ModelWrapper
from abc import ABC,abstractmethod
from numpy import ndarray
class FaceClassifier(ABC,ModelWrapper):
    model = None

    def __init__(self, model_name: str):
        super().__init__(model_name)

    @abstractmethod
    def predict(self, X: ndarray) -> int:
        """Classifies an embedding vector into a cluster_id"""
        raise NotImplementedError
    
    @abstractmethod
    def train(self, X: ndarray, n_clusters: int, random_state: int = 42):
        """Trains a classifier on the given embeddings"""
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self, X: ndarray) -> dict:
        """Evaluates the classifier on the given embeddings, returning metrics"""
        raise NotImplementedError
    