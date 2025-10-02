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
    input_size = (112, 112)
    output_shape = (512,)

    def __init__(self,model_name: str):
        super().__init__(model_name)

    @abstractmethod
    def embed_face(self, image) -> list[float]:
        """Return an embedding vector (e.g. 512-D)"""
        raise NotImplementedError