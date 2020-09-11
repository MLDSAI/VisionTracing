import os

from flask import (
    Flask, abort, Blueprint, flash, render_template, redirect, request,
    url_for, flash, jsonify, send_from_directory, Markup
)
from loguru import logger
from rq import Queue
import rq_dashboard
from worker import conn, redis_url
import time
import json
from flask_socketio import SocketIO

q = Queue(connection=conn)


app = Flask(__name__, static_url_path='/static')
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
app.config['RQ_DASHBOARD_REDIS_URL'] = redis_url
app.config.from_object(rq_dashboard.default_settings)
app.register_blueprint(
    rq_dashboard.blueprint, url_prefix='/rq/'
)
jobs = []

socket = SocketIO(app, cors_allowed_origins='*',  message_queue=os.getenv('REDIS_URL')) 

@app.template_filter('refresh_job')
def refresh_job(job):
    '''
    This function updates the meta dictionary of a given job and returns the
    job's filename
    Parameters:
    - job: REDIS Queue job
    '''
    try:
        job.refresh()
        print('Job refreshed successfully')
    except Exception as e: 
        print('Job did not refresh properly, exception {}'.format(e))
    return job.filename

@app.template_filter('video_exists')
def video_exists(job):
    '''
    This function checks whether or not a video with the job's track filename 
    exists. If it exists, then it returns HTML markup for the video source.
    Else, it returns the progress information.
    Parameters:
    - job: REDIS Queue job
    '''
    source = 'static/videos/{}'.format(job.tracks_filename)
    if os.path.exists(source):
        return Markup("""
        <video width="320" height="240" controls>
          <source src="/{}" type="video/mp4">
        </video>
        """.format(source))
    if job.meta.get('status'): 
        return job.meta.get('status')
    return 'Beginning process...'

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('', path)

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
