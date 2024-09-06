FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

# copying the application code and Poetry files
COPY ./app /app

COPY requirements.txt /app/

# installing dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt


# exposing the port FastAPI runs on
EXPOSE 8080

# command to run the FastAPI application (can be adjusted if needed)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
