import os

from flask import (
    Flask, abort, Blueprint, flash, render_template, redirect, request,
    url_for, flash, jsonify
)
from loguru import logger
from rq import Queue
import rq_dashboard

from worker import conn, redis_url


app = Flask(__name__)
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
app.config['RQ_DASHBOARD_REDIS_URL'] = redis_url
app.config.from_object(rq_dashboard.default_settings)
app.register_blueprint(
    rq_dashboard.blueprint, url_prefix='/rq/'
)


@app.route('/upload', methods=['POST'])
def upload():

    video_stream = request.files['video'].read()
    fpath_video = 'video.mp4'
    with open(fpath_video, 'wb') as f:
      f.write(video_stream)

    q = Queue(connection=conn)
    result = q.enqueue(
        'utils.convert_video_to_images', fpath_video
    )
    
    response = jsonify({'success': True})
    response.headers.add('Access-Control-Allow-Origin',  '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

  
@app.route('/test')
def test():
    logger.info('test()')
    q = Queue(connection=conn)
    result = q.enqueue(
        'utils.count_words_at_url', 'http://news.ycombinator.com'
    )
    logger.info(f'q: {q}')
    logger.info(f'result: {result}')
    return 'Enqueued'


@app.route('/')
def index():
    return render_template('index.html')
