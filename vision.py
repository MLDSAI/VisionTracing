import os
import shutil
import re
import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from loguru import logger
from tqdm import tqdm
from PIL import Image
import tracking
import time
import moviepy.video.io.ImageSequenceClip

def get_tracking_video(fpath_video, output_file):
    video, extension = fpath_video.split('.')
    logger.info(f'get_tracking_video fpath_video: {fpath_video}')
    image_gen  = _get_images_from_video(fpath_video)
    images = [image for image in image_gen]
    print("Number of frames in images is {}".format(len(images)))
    predictions = _get_predictions_from_images(images)
    tracks = tracking.get_tracks(predictions)
    fpath_tracking_video = _get_video_from_tracks(tracks, images, output_file)
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


def _get_video_from_tracks(tracks, images, output_file):
    ''' Save a video showing tracks to disk and return the path '''
    output_size = images[0].shape
    kelly_colors_rgb = [(255, 179, 0), (128, 62, 117), (255, 104, 0), (166, 189, 215),
                    (193, 0, 32), (206, 162, 98), (129, 112, 102), (0, 125, 52),
                    (246, 118, 142), (0, 83, 138), (255, 122, 92), (83, 55, 122),
                    (255, 142, 0), (179, 40, 81), (244, 200, 0), (127, 24, 13),
                    (147, 170, 0), (89, 51, 21), (241, 58, 19), (35, 44, 22)]

    kelly_colors = [[x[2], x[1], x[0]] for x in kelly_colors_rgb]
    
    image_folder = "image_folder_{}".format(time.time())    
    os.mkdir(image_folder)
    
    image_files = []
    for i in range(len(tracks[0])): # Number of tracks
        track_frame = np.zeros((output_size[0], output_size[1], 3), dtype=np.float32)

        for j in range(len(tracks)): # Number of bounding boxes within a track
            pt = tracks[j][i]
            if any(np.isnan(pt)):
                continue
            x1, y1, x2, y2 = pt
            x, y, w, h = x1, y1, x2 - x1, y2 - y1 # Top left coordinates and width and height respectively
            cv2.rectangle(track_frame, (int(x), int(y)), (int(x + w), int(y + h)),
                          kelly_colors[j % len(kelly_colors)], 2)
        
        frame = np.where(track_frame != 0, track_frame, images[i])
        frame = Image.fromarray(frame.astype(np.uint8))
        image_file = '{}/frame{}.jpg'.format(image_folder, i)
        image_files.append(image_file)
        frame.save(image_file) 
    
    
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=15)
   
    if not os.path.exists("videos"):
        os.mkdir("videos")
    clip.write_videofile('videos/' + output_file)
   
    try:
        shutil.rmtree(image_folder)
    except:
        print("Error in deleting image folder")
    return output_file
