import os
import sys

from flask import Flask
from rq import Queue

from worker.worker import conn
from utils import count_words_at_url


app = Flask(__name__)
# TODO: read from environment
# XXX DEBUG -> INFO
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
# TODO: get rid of this
app.config['TIMEZONE'] = os.getenv(
  'TIMEZONE', 'America/Toronto'
)


def log(*args, **kwargs):
  print('app.py:', *args, **kwargs)
  sys.stdout.flush()


@app.route('/')
def index():
  log('index()')
  q = Queue(connection=conn)
  result = q.enqueue(count_words_at_url, 'http://heroku.com')
  log('q:', q)
  log('result:', result)
  return 'Enqueued'
