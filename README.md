# speech-to-sql

### How to run

Start the docker app on your laptop. Then run the following commands in a terminal.

```
cd frontend

docker build -t speech2sql:latest .

docker run -p 8000:8000 speech2sql

```

Upon successfull initiation of the container, you can navigate to 127.0.0.1:8000 in a browser and test out the application. 