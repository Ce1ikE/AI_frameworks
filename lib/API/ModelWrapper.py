from abc import ABC, abstractmethod

class ModelWrapper(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_name(self) -> str:
        return self.model_name
    
    @abstractmethod
    def settings(self) -> dict:
        """Return model settings as a dictionary."""
        raise NotImplementedError