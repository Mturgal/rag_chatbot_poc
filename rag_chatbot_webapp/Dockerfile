FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y gcc
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 80

CMD ["python", "webapp_novartis.py"]

