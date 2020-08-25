FROM python:3.7

# build time
ARG FLASK_ENV=production
# run time
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app
RUN pip3 install -r web.requirements.txt && pip install gunicorn
CMD bash /app/web.sh
