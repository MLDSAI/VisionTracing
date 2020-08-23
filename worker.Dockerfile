FROM python:3.7

ARG FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
CMD python worker.py
