# Databricks notebook source
# MAGIC %pip install git+https://github.com/sebrahimi1988/databricks-model-serving

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

#import mlflow
from databricks.model_serving.client import EndpointClient

# get API URL and token for Model Serving
databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

#mlflow_client = mlflow.MlflowClient()
model_serving_client = EndpointClient(databricks_url, databricks_token)

# COMMAND ----------

# DBTITLE 1,01. Serve the Model
# model_name = "falcon-7b-instruct-sepideh"
# endpoint_name = "databricks-llm-text2sql"

# model_version = mlflow_client.get_latest_versions(model_name, stages=["Production"])[
#     -1
# ].version
# models = [
#     {
#         "model_name": model_name,
#         "model_version": model_version,
#         "workload_type": "GPU_MEDIUM",
#         "workload_size": "Small",
#         "scale_to_zero_enabled": False,
#     }
# ]
# model_serving_client.create_inference_endpoint(endpoint_name, models)

# COMMAND ----------

# DBTITLE 1,03. Real-Time Inference with Prompt
# question = "How many distinct flights do we have in total?" # Good to go
# question = "How many bookings occurred in flight with id = 'FL001' on 2023-06-15?" # Good to go
# question = "How many bookings occurred in flight with id='FL001' yesterday?" # Good to go
# question = "Give me all the  booking ids for the 'JFKLAX' route" # Good to go
question = "Give me total number of bookings for each flight route?" # Good to go

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

prompt = f"""
You are a SQL expert. Given this table information:
'{data_details_instruction}'

Write a SQL query to answer the following question: '{question}'.

Return the resulting SQL query, without including any kind of explanation or comments.
"""

# COMMAND ----------

# question = "How many bookings occurred in flight with id='FL001' in the last 24 hours?" # Not Working
# question = "How many bookings occurred in flight with id='FL001' in the last 24 hours before midnight today?" # Not Working

# COMMAND ----------

input_data = {
  "inputs": {
    "prompt": [
      prompt
    ],
    "max_tokens": [
      1000
    ],
    "temperature": [
      0.1
    ]
  }
}

model_serving_client.query_inference_endpoint( "falcon-7b-instruct-sepideh", input_data)['predictions'].replace("\n", " ")#['candidates'][0]['text'][len(prompt):]
