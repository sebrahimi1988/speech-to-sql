# Databricks notebook source
!accelerate config default

# COMMAND ----------

from transformers import AutoTokenizer
import transformers
import torch
from datetime import datetime, timedelta

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map = "auto"
)

# COMMAND ----------

current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
sequences = pipeline(
   f"""
Assume you are a SQL specialist. Write a query that shows how many bookings there were last month, for each individual flight_id. Group by the results by flight_id. To answer this question. Consider the table flights, which has the following columns:
    - flight_id (int)
    - date (timestamp)

"""
,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# COMMAND ----------

current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
sequences = pipeline(
   f"""
Assume you are a SQL specialist. Write a sql query that shows how many bookings there were last month, for each individual flight_id. To answer this question. Consider the table flights, which has the following columns:
    - flight_id (int)
    - date (timestamp)

"""
,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

# COMMAND ----------


