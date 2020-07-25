import requests

from worker import log


def count_words_at_url(url):
  resp = requests.get(url)
  rval = len(resp.text.split())
  log('rval:', rval)
  return rval
