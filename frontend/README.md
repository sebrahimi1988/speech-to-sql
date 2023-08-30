# ASR Frontend

## Instructions


### Set up environmental variable.
Under the app folder, create a .env file with the following content:

```
BLOB_CONN_STR=<YOUR-BLOB-CONTAINER-CONNECTION-STRING>
ACCOUNT_URL=<YOUR-STORAGE-ACCOUNT-URL>
DATABRICKS_TOKEN=<YOUR-DATABRICKS-PAT-TOKEN> /* Where you are serving your models) */
DATABRICKS_URL=<YOUR-DATABRICKS-WORKSPACE-URL> /* Where you are serving your models) */
DATABRICKS_SQL_URL=<YOUR-DATABRICKS-WORKSPACE-URL> /* Where you have your data */
DATABRICKS_SQL_HTTP_PATH=<YOUR-SQL-WAREHOUSE-HTTP-PATH> 
DATABRICKS_SQL_TOKEN=<YOUR-DATABRICKS-PAT-TOKEN> /* Where you have your data */
```
### Build the Docker image and run it

Navigate to the frontend folder where the Dockerfile lives and run the following command

```
docker build -t text2sql:latest .
```

Once the build is complete run the following code to start a container

```
docker run -p 8000:8000 text2sql
```

In a web browser navigate to 127.0.0.1:8000 and try out the application.