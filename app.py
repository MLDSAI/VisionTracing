import os

from flask import (
    Flask, abort, Blueprint, flash, render_template, redirect, request,
    url_for, flash, jsonify, send_from_directory
)
from loguru import logger
from rq import Queue
import rq_dashboard

from worker import conn, redis_url
import time

q = Queue(connection=conn)


app = Flask(__name__, static_url_path='')
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
app.config['RQ_DASHBOARD_REDIS_URL'] = redis_url
app.config.from_object(rq_dashboard.default_settings)
app.register_blueprint(
    rq_dashboard.blueprint, url_prefix='/rq/'
)

jobs = []

@app.template_filter('job_refresh')
def job_refresh(job):
    try:
        job.refresh()
    except:
        pass
    return job.filename

@app.route('/videos/<path:path>')
def send_video(path):
    return send_from_directory('videos', path)

@app.route('/sample_video/<path:path>')
def send_sample_video(path):
    return send_from_directory('sample_video', path)

@app.route('/upload', methods=['POST'])
def upload():

    logger.info(f'upload() request.files: {request.files}')

    video_file = request.files.get('file')
    fname_video = video_file.filename
    video_stream = video_file.read()
    with open(fname_video, 'wb') as f:
      f.write(video_stream)
    
    one_week = 60 * 60 * 24 * 7
    fname, extension = fname_video.split('.')
    output_file = '{}-tracks{}.{}'.format(fname, time.time(), extension)
    
    job = q.enqueue(
        'vision.get_tracking_video',
        args=(fname_video, output_file),
        timeout=one_week
    )
    
    job.filename = fname_video
    job.tracks_filename = output_file 
    jobs.append(job)
    
    logger.info(f'job: {job}')
    
    return {
        'status': 200,
        'mimetype': 'application/json'
    }

  
@app.route('/test')
def test():
    logger.info('test()')
    one_day = 60 * 60 * 24
    result = q.enqueue(
        'utils.count_words_at_url',
        'http://news.ycombinator.com',
        job_timeout=one_day
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
