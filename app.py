from flask import Flask
from rq import Queue

from worker.worker import conn
from utils import count_words_at_url


app = Flask(__name__)


@app.route('/')
def index():
  q = Queue(connection=conn)
  result = q.enqueue(count_words_at_url, 'http://heroku.com')
  return 'Enqueued'
