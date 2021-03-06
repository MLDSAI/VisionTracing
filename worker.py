import os

import redis
from loguru import logger
from rq import Worker, Queue, Connection


listen = ['high', 'default', 'low']

redis_url = os.getenv('REDIS_URL')
print("Redis {}".format(redis_url))

conn = redis.from_url(redis_url)


if __name__ == '__main__':
  logger.info('__main__')
  with Connection(conn):
    worker = Worker(map(Queue, listen))
    logger.info(f'worker: {worker}')
    worker.work()
