import os
import sys

import redis
from loguru import logger
from rq import Worker, Queue, Connection


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
  logger.info('__main__')
  with Connection(conn):
    worker = Worker(map(Queue, listen))
    logger.info(f'worker: {worker}')
    worker.work()
