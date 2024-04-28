# syntax=docker/dockerfile:1.2
FROM python:3.12.3
# put you docker configuration here


RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt && pip install -r requirements-dev.txt && pip install -r requirements-test.txt


EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "api.main:app", "-w", "4", "--timeout", "300", "-k", "uvicorn.workers.UvicornWorker"]