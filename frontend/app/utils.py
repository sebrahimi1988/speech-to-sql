import os
import logging

from azure.storage.blob import BlobClient, BlobServiceClient
from databricks.model_serving.client import EndpointClient

from databricks import sql

from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level="DEBUG")

SAS_TOKEN = os.environ["BLOB_CONN_STR"]
ACCOUNT_URL = os.environ["ACCOUNT_URL"]
DATABRICKS_URL = os.environ["DATABRICKS_URL"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
DATABRICKS_SQL_URL = os.environ["DATABRICKS_SQL_URL"]
DATABRICKS_SQL_HTTP_PATH = os.environ["DATABRICKS_SQL_HTTP_PATH"]
DATABRICKS_SQL_TOKEN = os.environ["DATABRICKS_SQL_TOKEN"]

client = BlobServiceClient(account_url=ACCOUNT_URL, credential=SAS_TOKEN)


def upload_blob_data(
    file_path: str,
    container_name: str,
    blob_service_client: BlobServiceClient = client,
):
    file_name = file_path.split("/")[-1]

    blob_client = blob_service_client.get_blob_client(container="audio", blob=file_name)

    with open(file_path, "rb") as data:
        result = blob_client.upload_blob(data, blob_type="BlockBlob")
        logging.info(result)

    url = f"{ACCOUNT_URL}/audio/{file_name}"
    return url


def query_endpoint(audio_url: str) -> str:
    payload = {"inputs": {"audio_url": [audio_url]}}

    client = EndpointClient(DATABRICKS_URL, DATABRICKS_TOKEN)
    output = client.query_inference_endpoint(
        endpoint_name="whisper_sepideh", data=payload
    )

    return output["predictions"][0]


def query_llm_endpoint(question: str) -> str:
    """
    Helper method to wrap questions in a prompt and send them to the LLM to be translated to SQL.
    """
    data_details_instruction = """
    Table name: databricks_llm_pov.trading.bookings_with_flights

    Table description:

    - flight_id (string) 
    - booking_id (string) 
    - booking_date (date): The date when the booking was made.
    - load_factor (float)
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
    payload = {
        "inputs": {"prompt": [prompt], "max_tokens": [1000], "temperature": [0.1]}
    }

    client = EndpointClient(DATABRICKS_URL, DATABRICKS_TOKEN)
    output = client.query_inference_endpoint(
        endpoint_name="falcon-7b-instruct-sepideh", data=payload
    )

    sql_query = (
        "SELECT"
        + (
            output["predictions"]
            .replace("load_factor", "tickets_sold /  aircraft_capacity")
            .split(";")[0]
            + ";"
        ).split("SELECT")[1]
    )
    return sql_query



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
