import os

from flask import Flask
from rq import Queue

from worker.worker import conn
from utils import count_words_at_url


app = Flask(__name__)
# TODO: read from environment
# XXX DEBUG -> INFO
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
# TODO: get rid of this
app.config['TIMEZONE'] = os.getenv('TIMEZONE', 'America/Toronto')

from logger_setup import logger

logger.warning('*' * 40)
DEBUG = os.environ.get('DEBUG', 0)
logger.warning('DEBUG: {}'.format(DEBUG))
is_debug = bool(int(DEBUG))
logger.warning('is_debug: {}'.format(is_debug))
app.debug = is_debug
logger.warning('*' * 40)


@app.route('/')
def index():
  q = Queue(connection=conn)
  result = q.enqueue(count_words_at_url, 'http://heroku.com')
  return 'Enqueued'
