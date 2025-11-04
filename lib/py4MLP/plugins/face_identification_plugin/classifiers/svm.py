from .base import FaceClassifier, LearningType
from ..utils.model_backend import BackendType

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVMClassifier(FaceClassifier):
    def __init__(self, kernel="linear"):
        super().__init__(
            BackendType.SKLEARN, 
            LearningType.SUPERVISED
        )
        self.model = SVC(kernel=kernel)

    def train(self, embeddings, labels):
        self.model.fit(embeddings, labels)
        preds = self.model.predict(embeddings)
        self._metrics = {"training_accuracy": accuracy_score(labels, preds)}

    def predict(self, embedding):
        return self.model.predict([embedding])[0]