import numpy as np
import torch
import librosa
import pandas as pd
import os
import h5py
from glob import glob
from tqdm import tqdm
import kaldi_io
import pypeln.process as pr
import argparse
import models as Model
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str)
parser.add_argument('--config', type=str)
parser.add_argument('-p', '--path', type=str)
parser.add_argument('-w', '--win_length', type=int, default=40)
parser.add_argument('-o', '--output', type=str)
parser.add_argument('-t', '--type', type=str)
parser.add_argument('-c', type=int, default=20)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def raw_features(file):
    audio, sr = librosa.load(file)
    ID = os.path.split(file)[-1].split('_')[0]
    t_df = pd.read_csv(os.path.join(args.path, \
        "{}_TRANSCRIPT.csv".format(ID)), sep='\t')
    t_df = t_df[t_df.speaker=='Participant']
    start_end = t_df.loc[:, ['start_time', 'stop_time']].values
    win_length = int(args.win_length * sr / 1000)
    hop_length = win_length // 4
    features = []

    for start, end in start_end:
        if (args.type == 'stft'):
            feature = np.log(
                np.abs(
                    librosa.stft(
                        audio[int(start * sr):int(end * sr)], \
                        1023, win_length=win_length
                    )
                ).T + 1e-12
            )
        else:
            feature = np.log(
                librosa.feature.melspectrogram(
                audio[int(start * sr):int(end * sr)], hop_length=hop_length).T + 1e-12
            )
        features.append(feature)
        #features.extend(feature)
    return ID, features
    
files = glob(os.path.join(args.path, "*_AUDIO.wav"))

# with tqdm(total=len(files)) as pbar, \
#     open(args.output, 'wb') as output:
    
#     for ID, features in pr.map(raw_features, \
#         files, workers=args.c, maxsize=args.c*2):

#         kaldi_io.write_mat(output, np.array(features), str(ID))
#         pbar.update()


params = torch.load(args.model, map_location='cpu')
config = torch.load(args.config)['config']
model = getattr(Model, config['model'])(**config['model_args'])
model.load_state_dict(params)
model = model.to(device)

with tqdm(total=len(files)) as pbar, \
    torch.set_grad_enabled(False), \
    open(args.output, 'wb') as output:
    
    for ID, features in pr.map(raw_features, \
        files, workers=args.c, maxsize=args.c*2):

        out_mat = []
        for feature in features:
            feature = torch.from_numpy(feature).to(torch.float).to(device)
            out = model.extract_embedding(feature.unsqueeze(0)).squeeze(0).cpu()
            out_mat.append(out.numpy())
        kaldi_io.write_mat(output, np.array(out_mat), str(ID))
        pbar.update()

