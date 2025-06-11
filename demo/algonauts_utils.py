"""
This is a set of streamlined utilities for downloading and loading the stimuli and fMRI data from the algonauts 2021 dataset. 
These are mainly written for use in the full_pipeline_demo.ipynb notebook, and are not necessary for use of VWAM on other datasets.
"""

import os
import pickle
import numpy as np
import requests
import zipfile
import io
import cv2
import glob

def download_fmri():
    """Download the full fMRI dataset"""
    print("Downloading fMRI dataset...")
    dropbox_link = 'https://www.dropbox.com/s/agxyxntrbwko7t1/participants_data.zip?dl=1'
    
    # Download and extract the full dataset
    response = requests.get(dropbox_link)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall()
    print("fMRI dataset downloaded and extracted")

def download_videos():
    """Download the video dataset"""
    print("Downloading videos...")
    video_url = "https://www.dropbox.com/s/agxyxntrbwko7t1/AlgonautsVideos268_All_30fpsmax.zip?dl=1"
    
    # Download and extract videos
    response = requests.get(video_url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall()
    print("Videos downloaded and extracted")

def load_roi_data(subject, roi, data_dir='./participants_data_v2021', download=True):
    """
    Load ROI data for a subject.
    Args:
        subject: str, subject ID (e.g., 'sub04')
        roi: str, ROI name (e.g., 'FFA', 'V1', etc.)
        data_dir: str, path to data directory
        download: bool, whether to download data if not found
    Returns:
        roi_data: np.ndarray (num_videos x num_voxels)
    """
    roi_path = os.path.join(data_dir, 'mini_track', subject, f'{roi}.pkl')
    
    # Check if file exists, download if needed
    if not os.path.exists(roi_path):
        if download:
            print(f"ROI data not found at {roi_path}")
            download_fmri()
        else:
            raise FileNotFoundError(f"ROI data not found at {roi_path} and download=False")
    
    with open(roi_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    return np.mean(data['train'], axis=1)  # average over repetitions

def load_video(video_idx, video_dir='./AlgonautsVideos268_All_30fpsmax', download=True):
    """
    Load a single video by its index (0-999).
    Args:
        video_idx: int, index of video to load (0-999)
        video_dir: str, path to directory containing videos
        download: bool, whether to download data if not found
    Returns:
        frames: np.ndarray (num_frames x height x width x channels)
    """
    # Check if directory exists
    if not os.path.exists(video_dir):
        if download:
            print(f"Video directory not found at {video_dir}")
            download_videos()
        else:
            raise FileNotFoundError(f"Video directory not found at {video_dir} and download=False")
    
    # Get list of videos and check index
    video_list = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    if video_idx >= len(video_list):
        raise ValueError(f"Video index {video_idx} out of range. Must be 0-{len(video_list)-1}")
    
    # Load the video
    cap = cv2.VideoCapture(video_list[video_idx])
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return np.array(frames)