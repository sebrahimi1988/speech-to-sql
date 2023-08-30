FROM python:3.10.12-slim-buster

RUN apt update -y && apt install git -y

RUN mkdir /app && mkdir /files && mkdir /app/files

ENV UPLOAD_FOLDER="/tmp"

WORKDIR /app

COPY app .
COPY requirements.txt ./requirements.txt

RUN pip install -U pip && pip install -r requirements.txt

CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8000"]
