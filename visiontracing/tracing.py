def _get_images_from_videos(dirpath_videos):
    '''
    Parameters:
    - list[str] dirpath_videos: path to directory/bucket containing video files

    Return:
    {
        '<path_to_video>': [np.ndarray, ...],
        ...
    }
    '''

    images_by_fpath_video = {}

    # TODO

    return images_by_fpath_video


def _get_tracks_from_images(images_by_fpath_video):
    '''
    Parameters:
    - {str: list[np.ndarray]} images: list of images in chronological order,
      keyed by path to video

    Return:
    - TODO
    '''

    # TODO
    pass


def _get_subjects_from_tracks(tracks_by_fpath_video):
    '''
    Parameters:
    - TODO

    Return:
    - TODO
    '''

    # TODO
    pass



def get_subjects(dirpath_videos):
    '''
    Get subjects from videos.

    Subjects are unique identifiers corresponding to the same person appearing
    in multiple images taken from multiple cameras.

    Parameters:
    - str dirpath_videos: path to directory containing videos
    '''

    images_by_fpath_video = _get_images_from_videos(dirpath_videos)
    tracks_by_fpath_video = _get_tracks_from_images(images_by_fpath_video)
    subjects = _get_subjects_from_tracks(tracks_by_fpath_video)
    return subjects
