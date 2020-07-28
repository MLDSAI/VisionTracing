import os

from flask import Flask
from loguru import logger
from rq import Queue
import rq_dashboard

from worker import conn, redis_url


app = Flask(__name__)
app.config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', 'DEBUG')
app.config['RQ_DASHBOARD_REDIS_URL'] = redis_url
app.config.from_object(rq_dashboard.default_settings)
app.register_blueprint(
  rq_dashboard.blueprint, url_prefix="/rq/"
)


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
