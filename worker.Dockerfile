FROM python:3.7
ARG FLASK_ENV=production
WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
CMD python worker.py
