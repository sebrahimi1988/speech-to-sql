# Databricks notebook source
# MAGIC %pip install git+https://github.com/huggingface/transformers.git@main

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizerFast
import mlflow
import torch

# COMMAND ----------

import huggingface_hub
#skip this if you are already logged in to hugging face
huggingface_hub.login()

# COMMAND ----------

model = "codellama/CodeLlama-7b-Instruct-hf"
repository = huggingface_hub.snapshot_download(repo_id=model, ignore_patterns="*.safetensors*")

# COMMAND ----------

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """"""

# Define PythonModel to log with mlflow.pyfunc.log_model

class Llama2(mlflow.pyfunc.PythonModel):
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
            pad_token_id=self.tokenizer.eos_token_id)
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        return f"""<s>[INST]<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n\n{instruction}[/INST]\n"""

    def _generate_response(self, prompt, temperature, max_new_tokens):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response
      
    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """

        outputs = []

        for i in range(len(model_input)):
          prompt = model_input["prompt"][i]
          temperature = model_input.get("temperature", [1.0])[i]
          max_new_tokens = model_input.get("max_new_tokens", [100])[i]

          outputs.append(self._generate_response(prompt, temperature, max_new_tokens))
      
        return outputs

# COMMAND ----------

with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Llama2(),
        artifacts={'repository' : repository},
        pip_requirements=["torch", "transformers", "accelerate"],
        input_example=pd.DataFrame({"prompt":["what is ML?"],"max_tokens": [80]}),
        registered_model_name='codellama-sepideh'
    )

# COMMAND ----------

# Load the logged model
import mlflow
model_name = 'codellama-sepideh'
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/1")

# COMMAND ----------

# Make a prediction using the loaded model

# question = "How many bookings were made in total?"
# question="How many bookings occurred in flight with id='FL001' today?"
question = "How many bookings occurred in flight with id='FL001' in the last 24 hours?" 
data_details_instruction = """
Table name: databricks_llm_pov.trading.bookings_with_flights

Table description:

- flight_id (string) 
- booking_id (string) 
- booking_date (date): The date when the booking was made.
- number_of_passengers (int)
- days_prior_to_departure (int)
- departure_datetime (timestamp): The date and time when the flight is scheduled to depart.
- arrival_datetime (timestamp): The date and time when the flight is scheduled to arrive.
- aircraft_capacity (int) 
- total_ticket_available (int)
- tickets_sold (int)
- flight_route (string)
"""

prompt = f"""You are a SQL expert. Given this table information:
'{data_details_instruction}'

# Write a SQL query to answer the following question: '{question}'.

# Return the resulting SQL query, without including any kind of explanation or comments."""
input_example = pd.DataFrame({"prompt":[prompt],"max_tokens": [80]})
loaded_model.predict(input_example)

# COMMAND ----------


