FROM my-langchain-base:latest

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app


COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

# API listens on 8000; Chainlit UI on 8001
EXPOSE 8000 8001

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
