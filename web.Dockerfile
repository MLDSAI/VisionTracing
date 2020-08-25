# TODO: undo opencv business

FROM jjanzic/docker-python3-opencv

# build time
ARG FLASK_ENV=production
# run time
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt && pip install gunicorn
RUN pip3 install -e git://github.com/facebookresearch/detectron2.git#egg=detectron2
CMD bash /app/web.sh
