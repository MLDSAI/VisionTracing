import os
import sys

from flask import Flask
from loguru import logger
from rq import Queue

from worker.worker import conn


app = Flask(__name__)
# TODO: read from environment
# XXX DEBUG -> INFO
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
# TODO: get rid of this
app.config['TIMEZONE'] = os.getenv(
  'TIMEZONE', 'America/Toronto'
)


@app.route('/')
def index():
  logger.debug('index()')
  q = Queue(connection=conn)
  result = q.enqueue('utils.count_words_at_url', 'http://heroku.com')
  logger.debug(f'q: {q}')
  logger.debug(f'result: {result}')
  return 'Enqueued'
