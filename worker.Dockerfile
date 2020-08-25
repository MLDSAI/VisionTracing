FROM jjanzic/docker-python3-opencv

ARG FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
RUN pip3 install -e git://github.com/facebookresearch/detectron2.git#egg=detectron2
CMD jjanzic/docker-python3-opencv python worker.py
