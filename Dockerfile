FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dit -r requirements.txt

COPY main.py .

CMD ["python", "main.py"]
