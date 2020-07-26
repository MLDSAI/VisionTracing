import os

from flask import Flask
from loguru import logger
from rq import Queue

from worker.worker import conn


app = Flask(__name__)
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')


@app.route('/')
def index():
  logger.info('index()')
  q = Queue(connection=conn)
  result = q.enqueue('utils.count_words_at_url', 'http://news.ycombinator.com')
  logger.info(f'q: {q}')
  logger.info(f'result: {result}')
  return 'Enqueued'
