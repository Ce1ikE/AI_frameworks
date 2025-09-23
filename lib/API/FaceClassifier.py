from .ModelWrapper import ModelWrapper
from abc import ABC,abstractmethod


class FaceClassifier(ModelWrapper):
    def __init__(self, model_name: str):
        super().__init__(model_name)
    
    @abstractmethod 
    def classify_face(self, image_path: str) -> str:
        """Classifies an image based on embedding vector"""
        raise NotImplementedError
    
    @abstractmethod
    def train_classifier(self, embeddings: list, labels: list):
        """Trains a classifier on the given embeddings and labels"""
        raise NotImplementedError
    
    @abstractmethod
    def save_model(self, model_path: str):
        """Saves the trained classifier model to the specified path"""
        raise NotImplementedError
    
    @abstractmethod
    def load_model(self, model_path: str):
        """Loads a trained classifier model from the specified path"""
        raise NotImplementedError