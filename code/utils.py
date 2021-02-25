import torch
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
from loguru import logger
import sys
import yaml
from audiomentations import Compose,\
    AddGaussianNoise, TimeStretch, PitchShift, Shift

def gen_logger(file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if file:
        logger.add(file, enqueue=True, format=log_format)
    return logger

def get_config(config):
    with open(config) as config_file:
        config = yaml.load(config_file)
    return config

def get_index(h5, debug=False):
    with h5py.File(h5, 'r') as input:
        keys = list(input.keys())

    train, dev = train_test_split(keys, train_size=0.8)
    if debug:
        train, dev = train[:10], dev[:10]

    return train, dev

def process_fn(type='stft', p=0.5, sr=22050):
    augment_fn = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=p),
        PitchShift(min_semitones=-4, max_semitones=4, p=p),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=p)])
    win_length = int(20 * sr / 1000)
    if type == 'stft':
        def stft_transform(audio):
            audio = augment_fn(samples=audio, sample_rate=sr)
            features = np.log(np.abs(librosa.stft(
                audio, 1023, win_length=win_length)).T + 1e-12)
            return features
        return stft_transform
    if type == 'lms':
        def lms_transform(audio):
            audio = augment_fn(samples=audio, sample_rate=sr)
            hop_length = win_length // 5
            features = np.log(np.abs(librosa.feature.melspectrogram(
                audio, hop_length=hop_length)).T + 1e-12)
            return features
        return lms_transform



