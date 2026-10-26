FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects port 7860; other hosts can override via PORT.
ENV PORT=7860 MAX_UPLOAD_MB=200
EXPOSE 7860
CMD gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT
