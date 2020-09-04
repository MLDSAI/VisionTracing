# FROM jjanzic/docker-python3-opencv
FROM nvidia/cuda:11.0-runtime-ubuntu18.04

RUN apt-get update
RUN apt-get install python3.6 -y
RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y 

ARG FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app

RUN pip3 install --upgrade pip
RUN pip3 install PyYAML

RUN pip3 install numpy

RUN apt-get update
RUN apt-get install git -y
RUN pip3 install opencv-python
RUN pip3 install --upgrade cython && pip3 install -r worker.requirements.txt && python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y

CMD python3 worker.py
