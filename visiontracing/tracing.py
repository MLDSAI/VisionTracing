import os
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode

import torch
import numpy as np

def setup_cfg(config, opts, conf_thresh):
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


def _get_images_from_videos(video_stream):
    '''
    Parameters:
    - list[str] dirpath_videos: path to directory/bucket containing video files
    '''
    def _frame_from_video(video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break
    
    video = _frame_from_video(cv2.VideoCapture(video_stream))
    return video


def _get_tracks_from_images(video_file):
    '''
    Parameters:
    - {str: list[np.ndarray]} images: list of images in chronological order,
      keyed by path to video

    Return:
    - TODO
    '''
    cfg_name = 'keypoint_rcnn_R_50_FPN_3x.yaml' 
    DEFAULT_CONFIG = f'detectron2/configs/COCO-Keypoints/{cfg_name}'
    DEFAULT_CONF_THRESH = 0.1
    DEFAULT_OPTS = ['MODEL.WEIGHTS', model_zoo.get_checkpoint_url(f'COCO-Keypoints/{cfg_name}')]

    cfg = setup_cfg(DEFAULT_CONFIG, DEFAULT_OPTS, DEFAULT_CONF_THRESH)
    predictor = DefaultPredictor(cfg)

    all_predictions = []
    frames = []
    frame_gen = _get_images_from_videos("realshort.mp4")
    for i, frame in enumerate(frame_gen):
        frames.append(frame)
        predictions = predictor(frame)
        all_predictions.append(predictions)

    all_predictions = np.array(all_predictions)
    return all_predictions