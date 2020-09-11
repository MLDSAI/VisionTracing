FROM jjanzic/docker-python3-opencv

ARG FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app
RUN pip install --upgrade cython && pip install -r worker.requirements.txt && python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
CMD python worker.py
