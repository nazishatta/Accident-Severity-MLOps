FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt streamlit requests
COPY app/ ./app/
COPY models/ ./models/
COPY streamlit_app.py .
COPY start.sh .
RUN chmod +x start.sh
EXPOSE 7860
CMD ["./start.sh"]
