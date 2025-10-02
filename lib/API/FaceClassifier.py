from .ModelWrapper import ModelWrapper
from abc import ABC,abstractmethod
from numpy import ndarray

class FaceClassifier(ModelWrapper):
    model = None

    def __init__(self, model_name: str):
        super().__init__(model_name)

    @abstractmethod
    def predict(self, embedding: ndarray) -> str:
        """Return a label for the given embedding"""
        raise NotImplementedError
