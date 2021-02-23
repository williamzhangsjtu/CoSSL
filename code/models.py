import torch
import torch.nn as nn
import numpy as np

class encoder(nn.Module):
    """
    extract representation from septrogram
    most important part
    """
    def __init__(self, dim=128):

        out_channels, kernels, strides, padding = [4,16,64,dim], \
            [3,3,3,3], [2,2,2,2], [1,1,1,1]
        super(encoder, self).__init__()
        conv_part = nn.ModuleList()
        in_channel = 1
        for idx, kernel in enumerate(kernels):
            out_channel = out_channels[idx]
            conv_part.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 
                        kernel, strides[idx], padding[idx]),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(), nn.AvgPool2d(3, 1, padding=1)
                )
            )
            in_channel = out_channels[idx]
        conv_part.append(nn.AdaptiveAvgPool2d(1))

        self.ConvNet = nn.Sequential(*conv_part)
        self.out_dim = out_channels[-1]

    def forward(self, x):
        """
        x: B x T x D
        """
        x = x.unsqueeze(1)
        return self.ConvNet(x).flatten(1, 3)

class CoSSL(nn.Module):
    def __init__(self, dim=128, cap_Q=2048, topK=5):
        """
        TODO: mask out the same audio
        """
        super(CoSSL, self).__init__()

        self.register_buffer('MoCoQueue', torch.randn(dim, cap_Q))
        self.MoCoQueue = nn.functional.normalize(self.MoCoQueue, dim=0)
        self.register_buffer('QueuePtr', torch.zeros(1, dtype=torch.long))

        self.register_buffer('RefQueue', torch.randn(2304, cap_Q))
        self.RefQueue = nn.functional.normalize(self.RefQueue, dim=0)
        self.register_buffer('IndexQueue', torch.ones(cap_Q) * -1)
        
        self.queue_full = False
        self.topK = topK
        
        self.encoder = encoder(dim)

    @torch.no_grad()
    def _enqueue(self, keys, ref_keys, indices):
        B = keys.shape[0]
        ptr = self.QueuePtr[0]
        assert self.MoCoQueue.shape[1] % B == 0
        self.MoCoQueue[:, ptr: ptr+B] = keys.T
        self.RefQueue[:, ptr: ptr+B] = ref_keys.T
        self.IndexQueue[ptr: ptr+B] = indices

        ptr = (ptr + B) % self.MoCoQueue.shape[1]
        self.QueuePtr[0] = ptr
        if not ptr: self.queue_full = True

    def forward(self, feats, ref_feats, indices):
        q = self.encoder(feats[:, 1]) # B x D
        q = nn.functional.normalize(q, dim=1)
        ref_feats = nn.functional.normalize(ref_feats, dim=1)

        with torch.no_grad():
            k = self.encoder(feats[:, 0]) # B x D
            k = nn.functional.normalize(k, dim=1)

        score_pos = torch.einsum('bd,bd->b', [q, k]).unsqueeze(-1) # B x 1
        score_neg = torch.einsum('bd,dk->bk', [q, self.MoCoQueue.clone().detach()])

        mask_idx = indices.unsqueeze(1) == self.IndexQueue.unsqueeze(0)
        mask = mask_idx.to(torch.long)
        # pick topK samples as positive sample

        if self.queue_full:
            score_ref = torch.einsum('bd,dk->bk', 
                [ref_feats, self.RefQueue.clone().detach()])
            score_ref[mask.to(torch.bool)] = - np.inf
            _, idx = torch.topk(score_ref, self.topK, dim=1)
            weighted_mask = torch.ones_like(score_neg) * -1
            score_ref[mask.to(torch.bool)] = 1
            mask.scatter_(1, idx, 1)
            weighted_mask.scatter_(1, idx, 1)
            score_neg *= score_ref * weighted_mask

        score = torch.cat([score_pos, score_neg], dim=1)
        mask = torch.cat([torch.ones(q.shape[0], 1).to(mask.device), mask],dim=1)

        if q.requires_grad: self._enqueue(k, ref_feats, indices)

        
        return score, mask.to(score.device)

        
