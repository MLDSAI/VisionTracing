FROM python:3.7

# build time
ARG FLASK_ENV=production
# run time
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt && pip install gunicorn
CMD bash /app/web.sh

#FROM python:3.7-slim-buster
#ENV PYTHONDONTWRITEBYTECODE 1
#ENV PYTHONUNBUFFERED 1

#ADD container/system-packages.sh setup.py requirements.txt /app/
#ADD container/entrypoint.sh /usr/local/bin/
#RUN /app/system-packages.sh && \
#    rm /app/system-packages.sh && \
#    pip install gunicorn

#ADD visiontracing/ /app/visiontracing
#COPY VERSION /app/
#RUN pip install /app
#WORKDIR /app

#ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
#CMD ["/usr/local/bin/gunicorn", "--log-file=-", "--worker-tmp-dir=/dev/shm", "--workers=2", "--threads=4", "--worker-class=gthread", "--bind=0.0.0.0:8000", "visiontracing.app:api"]
