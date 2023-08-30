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
    render_template
)
import logging
import glob
from utils import upload_blob_data, query_endpoint, query_llm_endpoint, send_query_to_warehouse, save_to_table
import json

logging.basicConfig(level = "DEBUG")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.environ["UPLOAD_FOLDER"]
app.secret_key = "MY_SECRET_KEY" #Obs.: for dev purposes

@app.route('/')
def root():
    session["audio_file"] = ""
    session["question_text"] = ""
    session["question_sql"] = ""
    return render_template('index.html')

@app.route('/transcribed')
def transcribed():
    return render_template('index.html', transcribed = session["transcribed"])

@app.route('/audio', methods=['GET'])
def get_audio():

    files = glob.glob(f"{app.config['UPLOAD_FOLDER']}/*.mp3")
    logging.info(f"MP3 files: {files}")

    path = request.args.to_dict()["audio_path"]
    return send_file(f"{app.config['UPLOAD_FOLDER']}/{path}", mimetype = "audio/mpeg")

@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part

    audio_file = session.get("audio_file", "")

    if audio_file != "":
        logging.info(f"Existing file: {audio_file}, removing...")
        os.remove(audio_file)

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    logging.info(f"Received file: {file}")

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    file_name = str(uuid.uuid4()) + ".mp3"
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)
    session["audio_file"] = full_file_name

    file_url = upload_blob_data(
        file_path = full_file_name,
        container_name = "audio"
    )

    transcribed_text = query_endpoint(
        audio_url = file_url
    )

    logging.info(f"Transribed: {transcribed_text['text']}")
    # set the session's question_text to the transcription 
    # call result to be sent to the text2sql model in the next stage
    session["question_text"] = transcribed_text['text']
    return transcribed_text['text']

@app.route('/set-question-text', methods=['POST'])
def set_question_text():    
    question_text = request.form['question_text']
    session["question_text"] = question_text
    save_to_table(question_text)
    return question_text

@app.route('/convert-text2sql', methods=['POST'])
def convert_text2sql():
    """
    takes the question in natural language and sends it to the LLM for conversion to SQL
    """ 

    question_text = session.get("question_text", "")

    # if question_text != "":
    #     logging.info(f"Existing text: {question_text}, removing...")
    #     #os.remove(audio_file)

    # if 'question' not in request.form.items:
    if question_text.strip() == "":
        flash('No question part')
        return redirect(request.url)

    # question_text = request.form['question']
    logging.info(f"Received question: {question_text}")

    sql_query = query_llm_endpoint(question_text)
    session["question_sql"] = sql_query
    return sql_query

@app.route('/set-question-sql', methods=['POST'])
def set_question_sql():    
    question_sql = request.form['question_sql']
    session["question_sql"] = question_sql
    return question_sql

@app.route('/send-sql-query', methods=['POST'])
def send_sql_query():
    """
    sends the sql query to a SQL Warehouse and returns result
    """ 

    question_sql = session.get("question_sql", "")

    if question_sql.strip() == "":
        flash('No question part')
        return redirect(request.url)

    # question_text = request.form['question']
    logging.info(f"Sending query: {question_sql}")

    sql_result = send_query_to_warehouse(question_sql)
    
    return sql_result


if __name__ == '__main__':
    app.run()