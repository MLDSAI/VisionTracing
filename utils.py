import requests

from loguru import logger

from visiontracing.tracing import _get_images_from_videos



def count_words_at_url(url):
    resp = requests.get(url)
    rval = len(resp.text.split())
    logger.info(f'rval: {rval}')
    return rval


def convert_video_to_images(fpath_video):
    logger.info(f'convert_video_to_images() fpath_video: {fpath_video}')
    images = list(_get_images_from_videos(cv2.VideoCapture('video.mp4')))
