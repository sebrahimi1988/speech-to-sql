# Databricks notebook source
from transformers import pipeline

generator = pipeline(task="automatic-speech-recognition", model = "openai/whisper-small")

# COMMAND ----------

generator("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

# COMMAND ----------

# DBTITLE 1,Simple Test
from models.whisper import WhisperModel

model = WhisperModel()

payload = {"audio_url": ["https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"]}
context = None
model.load_context(context)
model.predict(context = context, model_input = payload)

# COMMAND ----------

import mlflow

with mlflow.start_run() as run:
  model_info = mlflow.pyfunc.log_model(
      artifact_path = "model",
      python_model = WhisperModel(),
      code_path = ["../../models/whisper.py"],
      pip_requirements = [
      "transformers==4.28.1",
      "torch==1.13.1"
    ]
  )

# COMMAND ----------


