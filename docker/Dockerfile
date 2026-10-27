FROM python:3.11-slim

WORKDIR /app

COPY model/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model/ .

EXPOSE 5000

CMD ["python", "predict.py"]