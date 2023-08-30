# Databricks notebook source
# Databricks notebook source
#%pip install --upgrade torch
import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
print(torch.__version__)

# COMMAND ----------

from huggingface_hub import snapshot_download
# Download the Falcon model snapshot from huggingface
snapshot_location = snapshot_download(repo_id="tiiuae/falcon-7b-instruct",  ignore_patterns="coreml/*")

# COMMAND ----------

class Falcon(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], padding_side="left")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id).to('cuda')
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        return instruction

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"][0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [100])[0]

        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Falcon(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate", "einops","sentencepiece"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# Register model in MLflow Model Registry
model_name = "falcon-7b-instruct-sepideh"
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    model_name
)
# Note: Due to the large size of the model, the registration process might take longer than the default maximum wait time of 300 seconds. MLflow could throw an exception indicating that the max wait time has been exceeded. Don't worry if this happens - it's not necessarily an error. Instead, you can confirm the registration status of the model by directly checking the model registry. This exception is merely a time-out notification and does not necessarily imply a failure in the registration process.

# COMMAND ----------

# Load the logged model
# loaded_model = mlflow.pyfunc.load_model(f"models:/{result.name}/{result.version}")

# COMMAND ----------

# # Make a prediction using the loaded model
# # question = "How many bookings were made in total?" # Good to go
# # question = "How many bookings occurred in flight with id = 'FL001' on 2023-06-15?" # Good to go
# # question = "How many bookings occurred in flight with id='FL001' today?" # Good to go
# # question = "How many bookings occurred in flight with id='FL001' in the last 24 hours?" # Good to go
# # question = "Give me all the  booking ids for the 'JFKLAX' route" # Good to go
# question = "Give me total number of bookings for each flight route?" # Good to go

# data_details_instruction = """
# Table name: databricks_llm_pov.trading.bookings_with_flights

# Table description:

# - flight_id (string) 
# - booking_id (string) 
# - booking_date (date): The date when the booking was made.
# - number_of_passengers (int)
# - days_prior_to_departure (int)
# - departure_datetime (timestamp): The date and time when the flight is scheduled to depart.
# - arrival_datetime (timestamp): The date and time when the flight is scheduled to arrive.
# - aircraft_capacity (int) 
# - total_ticket_available (int)
# - tickets_sold (int)
# - flight_route (string)
# """

# prompt = f"""
# You are a SQL expert. Given this table information:
# '{data_details_instruction}'

# Write a SQL query to answer the following question: '{question}'.

# Return the resulting SQL query, without including any kind of explanation or comments.
# """

# input_example=pd.DataFrame({"prompt":[prompt], "temperature": [0.1],"max_tokens": [100]})
# loaded_model.predict(input_example)

# COMMAND ----------


