import requests

from loguru import logger


def count_words_at_url(url):
  resp = requests.get(url)
  rval = len(resp.text.split())
  logger.info(f'rval: {rval}')
  return rval