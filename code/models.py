import torch
import torch.nn as nn
import numpy as np
from efficientnet_pytorch import EfficientNet



def init_param(model):
    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.zeros_(param)
        else:
            torch.nn.init.normal_(param, std=10)

class efficientnet(nn.Module):
    def __init__(self, pretrain='efficientnet-b0'):
        super(efficientnet, self).__init__()
        self.encoder = EfficientNet.from_pretrained(pretrain)

    def forward(self, x):
        return self.encoder.extract_features(x).mean((2,3))

class Encoder(nn.Module):
    """
    extract representation from septrogram
    most important part
    """
    def __init__(self, dim=128):

        out_channels, kernels, strides, padding = [4,16,64,dim], \
            [3,3,3,3], [2,2,2,2], [1,1,1,1]
        # out_channels, kernels, strides, padding = [4,dim], [3,3], [2,2], [1,1]
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
        init_param(self.ConvNet)
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

        self.ifhard = False
        
        # self.encoder = encoder(dim)
        self.encoder = efficientnet()

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

    def EasyLearing(self, feats, indices):
        # B, H, W = feats.shape[0], feats.shape[2], feats.shape[3]
        # embeddings = self.encoder(feats.view(-1, H, W))
        # embeddings = nn.functional.normalize(
        #     embeddings.view(B, 2, -1), dim=2)
        # q, k = embeddings[:, 1], embeddings[:, 0]

        feats = feats.unsqueeze(1).repeat(1,3,1,1,1)
        q = self.encoder(feats[:, :, 1]) # B x D
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            k = self.encoder(feats[:, :, 0]) # B x D
            k = nn.functional.normalize(k, dim=1)
        s = torch.einsum('bd,dk->bk', [q, k.transpose(0, 1)])
        mask = indices.unsqueeze(1) == indices.unsqueeze(0)


        return s, mask.to(torch.long)


    def HardLearing(self, feats, ref_feats, indices):
        feats = feats.unsqueeze(1).repeat(1,3,1,1,1)
        q = self.encoder(feats[:, :, 1]) # B x D
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            k = self.encoder(feats[:, :, 0]) # B x D
            k = nn.functional.normalize(k, dim=1)

        # score_pos = torch.einsum('bd,bd->b', [q, k]).unsqueeze(-1) # B x 1
        score_batch = torch.einsum('bd,dk->bk', [q, k.transpose(0, 1)]) # B x B
        score_queue = torch.einsum('bd,dk->bk', [q, self.MoCoQueue.clone().detach()])

        # the first B elements in ref_indices are just indices
        mask_idx = indices.unsqueeze(1) == self.IndexQueue.unsqueeze(0)
        mask_queue = mask_idx.to(torch.long)

        # pick topK samples as positive sample
        if self.queue_full:
            score_ref = torch.einsum('bd,dk->bk', 
                [ref_feats, self.RefQueue.clone().detach()])
            tmp = score_ref.clone().detach()
            tmp[mask_queue.to(torch.bool)] = - np.inf
            _, idx = torch.topk(tmp, self.topK, dim=1)
            weighted_mask = torch.ones_like(score_queue) * -1
            mask_queue.scatter_(1, idx, 1)
            weighted_mask.scatter_(1, idx, 1)
            score_queue *= score_ref * weighted_mask

        score = torch.cat([score_batch, score_queue], dim=1)
        mask_batch = indices.unsqueeze(1) == indices.unsqueeze(0)
        mask = torch.cat([mask_batch, mask_queue],dim=1)

        if q.requires_grad: self._enqueue(k, ref_feats, indices)

        
        return score, mask.to(score.device)

    def forward(self, feats, ref_feats, indices):

        if self.ifhard:
            return self.HardLearing(feats, ref_feats, indices)
        return self.EasyLearing(feats, indices)

    def extract_embedding(self, feats):
        feats = feats.unsqueeze(1).repeat(1,3,1,1)
        emb = self.encoder(feats)
        return nn.functional.normalize(emb, dim=1)
