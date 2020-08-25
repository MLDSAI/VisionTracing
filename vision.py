import os

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from loguru import logger
from tqdm import tqdm

import tracking


def get_tracking_video(fpath_video):
    logger.info(f'get_tracking_video fpath_video: {fpath_video}')
    image_gen  = _get_images_from_video(fpath_video)
    images = [image for image in image_gen]
    predictions = _get_predictions_from_images(images)
    tracks = tracking.get_tracks(predictions)
    fpath_tracking_video = _get_video_from_tracks(tracks, images)
    return len(images), fpath_tracking_video


def _get_images_from_video(fpath_video):
    '''
    Parameters:
    - str fpath_video: path to video file
    '''

    logger.info(
        f'_get_images_from_video() fpath_video: {fpath_video}'
    )

    def _frame_from_video(_video_capture):
        while _video_capture.isOpened():
            retval, image = _video_capture.read()
            if retval:
                yield image
            else:
                break
    
    video_capture = cv2.VideoCapture(fpath_video)
    image_gen = _frame_from_video(video_capture)
    return image_gen


def _get_predictions_from_images(images):
    '''
    Parameters:
    - list[np.ndarray images: list of images in chronological order

    Return:
    - TODO
    '''

    dirpath_models = os.path.dirname(model_zoo.__file__)
    logger.info(
        f'_get_predictions_from_images() '
        'dirpath_models: {dirpath_models}'
    )

    fname_config = 'keypoint_rcnn_R_50_FPN_3x.yaml' 
    DEFAULT_CONFIG = os.path.join(
        dirpath_models, 'configs', 'COCO-Keypoints', fname_config
    )
    DEFAULT_CONF_THRESH = 0.1
    DEFAULT_OPTS = [
        'MODEL.WEIGHTS',
        model_zoo.get_checkpoint_url(
            f'COCO-Keypoints/{fname_config}'
        )
    ]
    cfg = _setup_cfg(
        DEFAULT_CONFIG, DEFAULT_OPTS, DEFAULT_CONF_THRESH
    )
    predictor = DefaultPredictor(cfg)
    predictions = []
    for i, image in enumerate(tqdm(images)):
        image_predictions = predictor(image)
        predictions.append(image_predictions)

    predictions = np.array(predictions)
    logger.info(
        '_get_predictions_from_images() '
        'predictions: {predictions}'
    )
    return predictions


def _setup_cfg(config, opts, conf_thresh):
    # load config from file and arguments
    cfg = get_cfg()
    if not torch.cuda.device_count():
        print('Running on CPU')
        cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(config)
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf_thresh
    cfg.freeze()
    return cfg


def _get_video_from_tracks(tracks, images):
    ''' Save a video showing tracks to disk and return the path '''
    
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    output_size = images[0].shape
    output_file = 'tracks.mp4'
    out = cv2.VideoWriter(output_file, fourcc, 25, output_size[:-1])
    
    for i in range(len(tracks)): # Number of tracks
        track_frame = np.zeros((output_size[0], output_size[1], 3), dtype=np.float32)
        for j in range(len(tracks[0])): # Number of bounding boxes within a track
            pt = tracks[i][j]
            if any(np.isnan(pt)):
                continue
            x1, y1, x2, y2 = pt
            x, y, w, h = x1, y1, x2 - x1, y2 - y1 # Top left coordinates and width and height respectively
            cv2.rectangle(track_frame, (int(x), int(y)), (int(x + w), int(y + h)),
                          (0, 255, 0), 2)
        frame = np.where(track_frame != 0, images[i], track_frame)
        out.write(frame.astype(np.uint8))

    out.release()
    return output_file
