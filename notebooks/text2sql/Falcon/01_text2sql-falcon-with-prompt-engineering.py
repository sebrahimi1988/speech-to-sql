# Databricks notebook source
!accelerate config default

# COMMAND ----------

# DBTITLE 1,01. Loading the Model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_ckpt = "tiiuae/falcon-7b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000
)

# COMMAND ----------

# DBTITLE 1,02. Prompt Engineering
# question = "How many bookings were made in total?" # Good to go
# question = "How many bookings occurred in flight with id = 'FL001' on 2023-06-15?" # Good to go
# question = "How many bookings occurred in flight with id='FL001' today?" # Good to go
# question = "How many bookings occurred in flight with id='FL001' in the last 24 hours?" # Good to go
# question = "Give me all the  booking ids for the 'JFKLAX' route" # Good to go
# question = "Give me total number of bookings for each flight route?" # Good to go

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

# DBTITLE 1,03. Inference
sql_query = pipe(prompt)[0]['generated_text'][len(prompt):]
print(sql_query)

# COMMAND ----------

# Temporary access token to connect to a SQL warehouse
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

def run_query(query):

  from databricks import sql

  databricks_hostname = "e2-demo-field-eng.cloud.databricks.com"
  databricks_http_path = "/sql/1.0/warehouses/ead10bf07050390f"

  with sql.connect(
    server_hostname = databricks_hostname,
    http_path       = databricks_http_path,
    access_token    = databricks_token
  ) as connection:

    with connection.cursor() as cursor:
      cursor.execute(query)
      result = cursor.fetchall()

      for row in result:
        print(row)

# COMMAND ----------

# Run the generated query against a SQL warehouse
run_query(sql_query)

# COMMAND ----------


