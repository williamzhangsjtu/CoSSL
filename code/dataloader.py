from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import h5py
import numpy as np
import torch


class MoCoDataset(Dataset):
    def __init__(self, audio_h5, process_fn=lambda x:x, names=None):
        self.size = 96
        self.audio = None
        self.audio_h5 = audio_h5
        self.process_fn = process_fn
        
        self.names = names
        if names is None:
            with h5py.File(audio_h5, 'r') as input:
                self.names = input.keys()
        
        self.pos2idx, self.init_pos = [], {}
        counter = 0
        with h5py.File(audio_h5, 'r') as input:
            for i, n in enumerate(names):
                n_seg = len(input[n]) // self.size
                self.pos2idx.extend([i] * n_seg)
                self.init_pos[i] = counter
                counter += n_seg



    def __getitem__(self, i):
        if self.audio is None:
            self.audio = h5py.File(self.audio_h5, 'r')
        index = self.pos2idx[i]
        name = self.names[index]

        init_pos = self.init_pos[index]
        feats = self.audio[name][(i - init_pos) * self.size:
            (i - init_pos + 1) * self.size]
        feats_1 = self.process_fn(feats.copy()) # T x D
        feats_2 = self.process_fn(feats.copy()) # T x D

        return torch.tensor(feats_1), torch.tensor(feats_2),\
            torch.zeros(1), torch.tensor(int(index)).to(torch.long)

    def __len__(self):
        return len(self.pos2idx)


class CoSSLDataset_2(Dataset):
    def __init__(self, audio_h5, ref_h5, process_fn=lambda x:x, index=None):
        self.audio = None
        self.ref = None
        self.audio_h5 = audio_h5
        self.ref_h5 = ref_h5
        self.process_fn = process_fn
        
        # intervals = [0]
        total = 0
        if index is None:
            with h5py.File(audio_h5, 'r') as input:
                self.index = input.keys()
                for k in input.keys():
                    total += len(input[k])
                    # intervals.append(intervals[-1] + len(input[k]))
        else:
            self.index = index
            with h5py.File(audio_h5, 'r') as input:
                for k in index:
                    total += len(input[k])
                    # intervals.append(intervals[-1] + len(input[k]))
        # self.intervals = intervals
        self.total = total

    def __getitem__(self, i):
        if self.audio is None:
            self.audio = h5py.File(self.audio_h5, 'r')
        if self.ref is None:
            self.ref = h5py.File(self.ref_h5, 'r')
            
        index = self.index[np.random.randint(0, len(self.index))]
        # pos1 = np.random.randint(0, len(self.ref[index]))
        pos1, pos2 = np.random.choice(len(self.ref[index]), 2)
        feats_1, feats_2, ref1, ref2 = self.audio[index][pos1],\
            self.audio[index][pos2], self.ref[index][pos1], self.ref[index][pos2]

        feats_1 = self.process_fn(feats_1) # T x D
        feats_2 = self.process_fn(feats_2) # T x D


        return torch.tensor(feats_1), torch.tensor(feats_2),\
            torch.tensor(ref1), torch.tensor(ref2), torch.tensor(int(index)).to(torch.long)

    def __len__(self):
        return self.total

    # def _find_pos(self, i):
    #     low, high = 0, len(self.index) - 1
    #     while low < high:
    #         if high - low <= 1:
    #             break
    #         mid = (high + low) // 2
    #         if self.intervals[mid] > i:
    #             low = mid + 1
    #         elif self.intervals[mid] < i:
    #             high = mid - 1
    #         else:
    #             low = mid
    #             break
    #     return low



class CoSSLDataset(Dataset):
    def __init__(self, audio_h5, ref_h5, process_fn=lambda x:x, index=None):
        self.audio = None
        self.ref = None
        self.audio_h5 = audio_h5
        self.ref_h5 = ref_h5
        self.process_fn = process_fn
        
        if index is None:
            with h5py.File(audio_h5, 'r') as input:
                self.index = input.keys()
        else:
            self.index = index

    def __getitem__(self, i):
        if self.audio is None:
            self.audio = h5py.File(self.audio_h5, 'r')
        if self.ref is None:
            self.ref = h5py.File(self.ref_h5, 'r')
            
        index = self.index[i]
        audio_feats = self.audio[index][()]
        ref_feats = self.ref[index][()]
        feats_1 = self.process_fn(audio_feats.copy()) # T x D
        feats_2 = self.process_fn(audio_feats.copy()) # T x D

        speaker_idx = index.split('_')[0]


        return torch.tensor(feats_1), torch.tensor(feats_2),\
            torch.tensor(ref_feats), torch.tensor(int(speaker_idx)).to(torch.long)

    def __len__(self):
        return len(self.index)


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    feats_1, feats_2, ref_feats, indices = zip(*batch)
    feats_1 = pad_sequence(feats_1, batch_first=True)
    feats_2 = pad_sequence(feats_2, batch_first=True)
    return torch.stack([feats_1, feats_2], dim=1),\
        torch.stack(ref_feats), torch.tensor(indices).to(torch.long)

# def collate_fn(batch):
#     batch.sort(key=lambda x: len(x[0]), reverse=True)
#     feats_1, feats_2, ref_1, ref_2, indices = zip(*batch)
#     feats_1 = pad_sequence(feats_1, batch_first=True)
#     feats_2 = pad_sequence(feats_2, batch_first=True)
#     return torch.stack([feats_1, feats_2], dim=1),\
#         torch.stack([torch.stack(ref_1), torch.stack(ref_2)], dim=1),\
#         torch.tensor(indices).to(torch.long)


def create_dataloader(audio_h5, ref_h5=None, process_fn=lambda x: x, index=None, **kwargs):
    kwargs.setdefault("batch_size", 32)
    kwargs.setdefault("num_workers", 4)
    kwargs.setdefault("shuffle", True)

    if ref_h5 is None:
        _dataset = MoCoDataset(audio_h5, process_fn, index)
        return DataLoader(_dataset, collate_fn=collate_fn, drop_last=True, **kwargs)
    _dataset = CoSSLDataset(audio_h5, ref_h5, process_fn, index)
    return DataLoader(_dataset, collate_fn=collate_fn, drop_last=True, **kwargs)



# class CoSSLDataset(Dataset):
#     def __init__(self, audio_h5, ref_h5, process_fn=lambda x:x, index=None):
#         self.audio = None
#         self.ref = None
#         self.audio_h5 = audio_h5
#         self.ref_h5 = ref_h5
#         self.process_fn = process_fn
#         self.size = 96
        
#         if index is None:
#             with h5py.File(ref_h5, 'r') as input:
#                 self.index = input.keys()
#         else:
#             self.index = index
        
#         self.pos2idx, self.init_pos = [], {}
#         counter = 0
#         with h5py.File(ref_h5, 'r') as input:
#             for i in self.index:
#                 n_seg = len(input[i])
#                 self.pos2idx.extend([i] * n_seg)
#                 self.init_pos[i] = counter
#                 counter += n_seg

        


#     def __getitem__(self, i):
#         if self.audio is None:
#             self.audio = h5py.File(self.audio_h5, 'r')
#         if self.ref is None:
#             self.ref = h5py.File(self.ref_h5, 'r')
#         index = self.pos2idx[i]
#         init_pos = self.init_pos[index]
#         ref_feats = self.ref[index][i - init_pos] # D
#         # audio_feats = self.audio[index][
#         #     (i - init_pos) * self.size: (i - init_pos + 1) * self.size]
#         audio_feats = self.audio[index][str(i - init_pos)]
#         audio_feats = audio_feats[:2000] # limit the length to 20s
#         feats_1 = self.process_fn(audio_feats.copy()) # T x D
#         feats_2 = self.process_fn(audio_feats.copy()) # T x D


#         return torch.tensor(feats_1), torch.tensor(feats_2),\
#             torch.tensor(ref_feats), torch.tensor(int(index)).to(torch.long)

#     def __len__(self):
#         return len(self.pos2idx)