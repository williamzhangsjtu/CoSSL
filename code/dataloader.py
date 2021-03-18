from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import h5py
import numpy as np
import torch


class CoSSLDataset(Dataset):
    def __init__(self, audio_h5, ref_h5, process_fn=lambda x:x, index=None):
        self.audio = None
        self.ref = None
        self.audio_h5 = audio_h5
        self.ref_h5 = ref_h5
        self.process_fn = process_fn
        
        if index is None:
            with h5py.File(ref_h5, 'r') as input:
                self.index = input.keys()
        else:
            self.index = index
        
        self.pos2idx, self.init_pos = [], {}
        counter = 0
        with h5py.File(ref_h5, 'r') as input:
            for i in self.index:
                self.pos2idx.extend([i] * len(input[i]))
                self.init_pos[i] = counter
                counter += len(input[i])



    def __getitem__(self, i):
        if self.audio is None:
            self.audio = h5py.File(self.audio_h5, 'r')
        if self.ref is None:
            self.ref = h5py.File(self.ref_h5, 'r')
        index = self.pos2idx[i]
        init_pos = self.init_pos[index]
        ref_feats = self.ref[index][i - init_pos] # D
        audio_feats = self.audio[index][str(i - init_pos)][0]
        audio_feats = audio_feats[:100000] # limit the length
        feats_1 = self.process_fn(audio_feats) # T x D
        feats_2 = self.process_fn(audio_feats) # T x D


        return torch.tensor(feats_1), torch.tensor(feats_2),\
            torch.tensor(ref_feats), torch.tensor(int(index)).to(torch.long)

    def __len__(self):
        return len(self.pos2idx)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    feats_1, feats_2, ref_feats, indices = zip(*batch)
    feats_1 = pad_sequence(feats_1, batch_first=True)
    feats_2 = pad_sequence(feats_2, batch_first=True)
    return torch.stack([feats_1, feats_2], dim=1),\
        torch.stack(ref_feats), torch.tensor(indices).to(torch.long)


def create_dataloader(audio_h5, ref_h5, process_fn, index=None, **kwargs):
    kwargs.setdefault("batch_size", 32)
    kwargs.setdefault("num_workers", 4)
    kwargs.setdefault("shuffle", True)

    _dataset = CoSSLDataset(audio_h5, ref_h5, process_fn, index)

    return DataLoader(_dataset, collate_fn=collate_fn, drop_last=True, **kwargs)


