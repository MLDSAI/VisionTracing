import os
import sys

import redis
from loguru import logger
from rq import Worker, Queue, Connection


def log(*args, **kwargs):
  logger.debug('worker/worker.py:', *args, **kwargs)
  #sys.stdout.flush()


listen = ['high', 'default', 'low']

redis_url = os.getenv(
  'REDISTOGO_URL',
  (
    'redis://redistogo:e4c2902d2013e9a8f3a7bcbcd5458058@'
    'pike.redistogo.com:11724/'
  )
)

conn = redis.from_url(redis_url)


if __name__ == '__main__':
  log('__main__')
  with Connection(conn):
    worker = Worker(map(Queue, listen))
    log('worker:', worker)
    worker.work()
