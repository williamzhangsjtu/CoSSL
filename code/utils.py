import torch
from functools import wraps
import random
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
        train, dev = train[:len(train) // 4], dev[:len(dev) // 4]

    return train, dev

def spec_augment(feats, time_warping_para=80, frequency_masking_para=32,
        time_masking_para=24, frequency_mask_num=2, time_mask_num=2):

    v = feats.shape[1]
    tau = feats.shape[0]
    time_masking_para = tau if tau < time_masking_para else time_masking_para
    frequency_masking_para = tau if tau < frequency_masking_para else frequency_masking_para
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v-f)
        feats[:, f0:f0+f] = 0

    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau-t)
        feats[t0:t0+t, :] = 0

    return feats


def process_fn(output='stft', spec_aug=False, p=0.5, sr=22050):
    augment_fn = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=p),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=p),
        PitchShift(min_semitones=-4, max_semitones=4, p=p),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=p)])

    win_length = int(20 * sr / 1000)
    if output == 'stft':
        def stft_transform(feats):
            if feats.ndim == 1:
                feats = augment_fn(samples=feats, sample_rate=sr)
                feats = np.log(np.abs(librosa.stft(
                    feats, 1023, win_length=win_length)).T + 1e-12)
            if spec_aug:
                feats = spec_augment(feats)
            return feats
        return stft_transform
    if output == 'lms':
        def lms_transform(feats):
            if feats.ndim == 1:
                feats = augment_fn(samples=feats, sample_rate=sr)
                hop_length = win_length // 4
                feats = np.log(np.abs(librosa.feature.melspectrogram(
                    feats, n_fft=win_length, hop_length=hop_length, 
                    win_length=win_length)).T + 1e-12)
            if spec_aug:
                feats = spec_augment(feats)
            return feats
        return lms_transform


def process_func(audio, output='stft', **kwargs):
    if audio.ndim == 1:
        if output == 'stft':
            feats = stft_transform(audio, **kwargs)
        else:
            feats = lms_transform(audio, **kwargs)
        return spec_augment(feats)
    return spec_augment(audio)

def raw_audio_process(transform_fn):
    augment_fn = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)])

    @wraps(transform_fn)
    def augment_audio(audio, **kwargs):
        sr = kwargs.setdefault('sr', 22050)
        n_win = kwargs.setdefault('n_win', 20)
        win_length = int(n_win * sr / 1000)

        audio = augment_fn(audio)
        return transform_fn(audio, win_length=win_length, 
            hop_length=win_length // 4)
    return augment_audio


@raw_audio_process
def lms_transform(audio, **kwargs):
    n_fft = kwargs['win_length']
    feats = np.log(np.abs(librosa.feature.melspectrogram(
        audio, n_fft=n_fft, **kwargs)).T + 1e-12)
    return feats


@raw_audio_process
def stft_transform(audio, **kwargs):
    feats = np.log(np.abs(librosa.stft(
        audio, 1023, **kwargs)).T + 1e-12)
    return feats
