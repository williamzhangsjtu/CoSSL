import kaldi_io
import pandas as pd
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dev', type=str)
parser.add_argument('-t', '--train', type=str)
parser.add_argument('-i', '--input', type=str, default='daic_stft_w20_i11.ark')
parser.add_argument('--dev_out', default='dev_stft_w20_i11.ark', type=str)
parser.add_argument('--train_out', default='train_stft_w20_i11.ark', type=str)

args = parser.parse_args()

dev_df = pd.read_csv(args.dev)
train_df = pd.read_csv(args.train)

dev_ids, train_ids = dev_df.Participant_ID.values, train_df.Participant_ID.values
dev_ids = list(map(lambda x:str(x), dev_ids))
train_ids = list(map(lambda x:str(x), train_ids))

with open(args.dev_out, 'wb') as dev, open(args.train_out, 'wb') as train:
    for ID, features in kaldi_io.read_mat_ark(args.input):
        if ID in dev_ids:
            kaldi_io.write_mat(dev, features, ID)
        elif ID in train_ids:
            kaldi_io.write_mat(train, features, ID)

