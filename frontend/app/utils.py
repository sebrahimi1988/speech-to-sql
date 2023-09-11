import os
import logging

from databricks import sql
import pandas as pd

import json
import requests
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level="DEBUG")

DATABRICKS_EP_URL = os.environ["DATABRICKS_EP_URL"]
DATABRICKS_EP_TOKEN = os.environ["DATABRICKS_EP_TOKEN"]
DATABRICKS_SQL_URL = os.environ["DATABRICKS_SQL_URL"]
DATABRICKS_SQL_HTTP_PATH = os.environ["DATABRICKS_SQL_HTTP_PATH"]
DATABRICKS_SQL_TOKEN = os.environ["DATABRICKS_SQL_TOKEN"]


def query_endpoint(
    dataset, ep_name="whisper-sepideh",url=DATABRICKS_EP_URL, databricks_token=DATABRICKS_EP_TOKEN
):
    url = f"{url}/serving-endpoints/{ep_name}/invocations"
    headers = {
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    }
    ds_dict = {"dataframe_split": dataset.to_dict(orient="split")}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()["predictions"][0]


def send_query_to_warehouse(query: str) -> str:
    with sql.connect(
        server_hostname=DATABRICKS_SQL_URL,
        http_path=DATABRICKS_SQL_HTTP_PATH,
        access_token=DATABRICKS_SQL_TOKEN,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
    return result
    # for row in result:
    # print(row)


def save_to_table(text: str, table_name="databricks_llm_pov.trading.questions") -> str:
    """
    Helper method to save incoming questions to a table.
    """

    with sql.connect(
        server_hostname=DATABRICKS_SQL_URL,
        http_path=DATABRICKS_SQL_HTTP_PATH,
        access_token=DATABRICKS_SQL_TOKEN,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} (question string, date timestamp)"
            )

            cursor.execute(f"INSERT INTO {table_name} VALUES ('{text}', NOW())")

            result = cursor.fetchall()
    return result
