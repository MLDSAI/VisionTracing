import os

from flask import (
    Flask, abort, Blueprint, flash, render_template, redirect, request, url_for, flash, jsonify
)

from loguru import logger
from rq import Queue
import rq_dashboard

from worker import conn, redis_url

from visiontracing.tracing import _get_images_from_videos


app = Flask(__name__)
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
app.config['RQ_DASHBOARD_REDIS_URL'] = redis_url
app.config.from_object(rq_dashboard.default_settings)
app.register_blueprint(
  rq_dashboard.blueprint, url_prefix="/rq/"
)

@app.route('/upload', methods=["POST"])
def upload():

    video_stream = request.files['video'].read()
    with open("video.mp4", 'wb') as f:
      f.write(video_stream)
    
    images = list(_get_images_from_videos(cv2.VideoCapture("video.mp4")))

    response = jsonify({"success": True})
    response.headers.add('Access-Control-Allow-Origin',  '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response
  
@app.route('/')
def index():
  logger.info('index()')
  q = Queue(connection=conn)
  result = q.enqueue(
    'utils.count_words_at_url', 'http://news.ycombinator.com'
  )
  logger.info(f'q: {q}')
  logger.info(f'result: {result}')
  return 'Enqueued'
