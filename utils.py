import requests

import cv2
from loguru import logger

from visiontracing.tracing import _get_images_from_video


def convert_video_to_images(fpath_video):
    logger.info(
        f'convert_video_to_images() fpath_video: {fpath_video}'
    )
    image_gen = _get_images_from_video(fpath_video)
    images = [image for image in image_gen]
    logger.info(
        f'convert_video_to_images() len(images): {len(images)}'
    )
    logger.info(f'convert_video_to_images() done')


def count_words_at_url(url):
    logger.info(f'count_words_at_url() url: {url}')
    resp = requests.get(url)
    rval = len(resp.text.split())
    logger.info(f'count_words_at_url() rval: {rval}')
    return rval
