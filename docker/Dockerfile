FROM python:3.11-slim

WORKDIR /app

COPY model/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model/predict.py model/model.pkl ./

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health/ready', timeout=3)"]

USER 10001

CMD ["waitress-serve", "--listen=0.0.0.0:5000", "--threads=4", "predict:app"]