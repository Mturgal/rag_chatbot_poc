FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .
WORKDIR rag_chatbot_webapp

EXPOSE 80

CMD ["python", "webapp_novartis.py"]

