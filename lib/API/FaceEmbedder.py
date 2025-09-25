# interface for the Pipeline class
# Implementations of this class only need to make sure to output
# a embedding vector in order to classify each image in nD space
# e.g.:
# - DLibEmbedder
# - DArcFaceEmbedder
# - FaceNetEmbedder

from .ModelWrapper import ModelWrapper
from abc import ABC,abstractmethod

class FaceEmbedder(ModelWrapper):
    def __init__(self,model_name: str):
        super().__init__(model_name)

    @abstractmethod
    def embed_face(self, image) -> list[float]:
        """Return an embedding vector (e.g. 128-D)"""
        raise NotImplementedError