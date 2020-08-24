FROM python:3.7

ARG FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app
RUN apt update
RUN apt install python3-opencv -y 
RUN pip3 install -r requirements.txt
RUN pip3 install -e git://github.com/facebookresearch/detectron2.git#egg=detectron2
CMD python worker.py