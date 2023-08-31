import torch
import numpy as np
import mlflow.pyfunc

from transformers import pipeline

class WhisperModel(mlflow.pyfunc.PythonModel):
    """
    Class to wrap Open AI Whisper model
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """

        self._pipeline = pipeline(task="automatic-speech-recognition", model = "openai/whisper-small")

    def predict(self, context, model_input):

        transcription = self._pipeline(model_input["audio"]),

        return transcription


def _load_pyfunc(data_path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return WhisperModel()