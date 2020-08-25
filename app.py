import os

from flask import (
    Flask, abort, Blueprint, flash, render_template, redirect, request,
    url_for, flash, jsonify
)
from loguru import logger
from rq import Queue
import rq_dashboard

from worker import conn, redis_url

q = Queue(connection=conn)


app = Flask(__name__)
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
app.config['RQ_DASHBOARD_REDIS_URL'] = redis_url
app.config.from_object(rq_dashboard.default_settings)
app.register_blueprint(
    rq_dashboard.blueprint, url_prefix='/rq/'
)

jobs = []


@app.route('/upload', methods=['POST'])
def upload():

    logger.info(f'upload() request.files: {request.files}')

    video_file = request.files.get('file')
    fname_video = video_file.filename
    video_stream = video_file.read()
    with open(fname_video, 'wb') as f:
      f.write(video_stream)

    one_day = 60 * 60 * 24
    job = q.enqueue(
        'utils.convert_video_to_images',
        fname_video,
        ttl=one_day
    )
    job.filename = fname_video
    jobs.append(job)
    logger.info(f'job: {job}')

    return {
        'status': 200,
        'mimetype': 'application/json'
    }

  
@app.route('/test')
def test():
    logger.info('test()')
    result = q.enqueue(
        'utils.count_words_at_url',
        'http://news.ycombinator.com'
    )
    logger.info(f'q: {q}')
    logger.info(f'result: {result}')
    return 'Enqueued'


@app.route('/')
def index():
    logger.info(f'jobs: {jobs}')
    # TODO: use websockets to update client without refresh
    return render_template(
        'index.html',
        jobs=jobs
    )
