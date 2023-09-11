import os
import uuid
from flask import (
    Flask,
    flash,
    request,
    redirect,
    url_for,
    send_file,
    session,
    render_template,
)
import logging
import glob
import numpy as np
import pandas as pd
from utils import (
    query_endpoint,
    send_query_to_warehouse,
    save_to_table,
)
import pydub

from dotenv import load_dotenv


load_dotenv()

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


TRANSCRIPTION_MODEL = os.environ["TRANSCRIPTION_MODEL"]
TEXT2SQL_MODEL = os.environ["TEXT2SQL_MODEL"]


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.environ["UPLOAD_FOLDER"]
app.secret_key = "MY_SECRET_KEY"  # Obs.: for dev purposes


@app.route("/")
def root():
    return render_template("index.html")


@app.route("/save-record", methods=["POST"])
def save_record():
    def convert_audio_to_np(f, normalized=True):
        """Audio file to numpy array"""
        a = pydub.AudioSegment.from_file(f, format="webm").set_frame_rate(16000)
        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1, 2))
        if normalized:
            return a.frame_rate, np.float32(y) / 2**15
        else:
            return a.frame_rate, y

    # check if the post request has the file part
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["file"]

    logging.info(f"Received file: {file}")

    # Convert audio object received from microphone to a numpy array expected by Whisper
    sampling_rate, file_nparray = convert_audio_to_np(file)

    payload = pd.DataFrame({"audio": file_nparray, "sampling_rate": sampling_rate})

    transcribed_text = query_endpoint(dataset=payload, ep_name=TRANSCRIPTION_MODEL)

    logging.info(f"Transribed: {transcribed_text}")
    return transcribed_text


@app.route("/convert-text2sql", methods=["POST"])
def convert_text2sql():
    """
    takes the question in natural language and sends it to the LLM for conversion to SQL
    """

    question_text = request.form["question_text"]
    # question_text = session.get("question_text", "")
    if question_text.strip() == "":
        flash("No question part")
        return ''

    # question_text = request.form['question']
    logging.info(f"Received question: {question_text}")

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

    Write a SQL query to answer the following question: '{question_text}'.

    Return the resulting SQL query, without including any kind of explanation or comments.
    """
    payload = pd.DataFrame({"prompt": [prompt], "max_tokens": [80]})
    sql_query = query_endpoint(payload, ep_name=TEXT2SQL_MODEL)
    return sql_query


@app.route("/send-sql-query", methods=["POST"])
def send_sql_query():
    """
    sends the sql query to a SQL Warehouse and returns result
    """

    question_sql = request.form["question_sql"]
    logging.info(f"This is the question sql in the app: {question_sql}")

    if question_sql.strip() == "":
        flash("No question part")
        return ''

    logging.info(f"Sending query: {question_sql}")

    sql_result = send_query_to_warehouse(question_sql)

    return sql_result


if __name__ == "__main__":
    app.run()
