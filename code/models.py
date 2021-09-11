import torch
import torch.nn as nn
import numpy as np
import random
from efficientnet_pytorch import EfficientNet



def init_param(model):
    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.zeros_(param)
        else:
            torch.nn.init.xavier_uniform_(param)

class MLP(nn.Module):
    def __init__(self, in_dim=1280, out_dim=256, n_hidden=1):
        super(MLP, self).__init__()
        mlp = nn.ModuleList()
        _in_dim = in_dim
        for i in range(n_hidden):
            next_dim = _in_dim // 2 if _in_dim < 256 else 256
            mlp.append(nn.Linear(_in_dim, next_dim))
            mlp.append(nn.ReLU())
            _in_dim = next_dim
        mlp.append(nn.Linear(_in_dim, out_dim))
        # mlp.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    """
    extract representation from septrogram
    most important part
    """
    def __init__(self, out_dim=256):

        out_channels = [4,16,64,128,256,512]
        super(Encoder, self).__init__()
        conv_part = nn.ModuleList()
        in_channel = 1
        for idx, kernel in enumerate(kernels):
            out_channel = out_channels[idx]
            conv_part.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 3, 2, 1),
                    nn.MaxPool2d(3, 1, padding=1),
                    nn.BatchNorm2d(out_channel), nn.ReLU()
                )
            )
            in_channel = out_channels[idx]
        conv_part.append(nn.AdaptiveAvgPool2d(1))

        self.ConvNet = nn.Sequential(*conv_part)
        # init_param(self.ConvNet)
        dim = out_channels[-1]

        self.mlp = MLP(in_dim=dim, out_dim=out_dim)

    def forward(self, x):
        """
        x: B x T x D
        """
        conv_out = self.ConvNet(x).flatten(1, 3)
        return self.mlp(conv_out)

    def extract_rep(self, x):
        conv_out = self.ConvNet(x).flatten(1, 3)
        return conv_out


class ReCLR(nn.Module):
    def __init__(self, audio_dim=128, ref_dim=512, out_dim=256, cap_Q=2048, topK=5, momentum=0.999):

        super(CoSSL, self).__init__()

        self.register_buffer('MoCoQueue', torch.randn(audio_dim, cap_Q))
        self.MoCoQueue = nn.functional.normalize(self.MoCoQueue, dim=0)
        self.register_buffer('QueuePtr', torch.zeros(1, dtype=torch.long))

        self.register_buffer('RefQueue', torch.randn(ref_dim, cap_Q))
        self.RefQueue = nn.functional.normalize(self.RefQueue, dim=0)
        self.register_buffer('IndexQueue', torch.ones(cap_Q) * -1)
        
        self.queue_full = False
        self.topK = topK
        self.m = momentum
        self.cap_Q = cap_Q

        self.ifhard = False
        
        self.audio_encoder_q = Encoder(out_dim)
        self.audio_encoder_k = Encoder(out_dim)
        self._set_key_encoder(self.audio_encoder_q, self.audio_encoder_k)

    @torch.no_grad()
    def _set_key_encoder(self, encoder_q, encoder_k):
        for param_q, param_k in zip(
                encoder_q.parameters(), encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self, encoder_q, encoder_k):
        for param_q, param_k in zip(
                encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = param_q.data * (1 - self.m) + param_k.data * self.m

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


    def forward(self, audio, ref, indices):

        B = len(ref)
        audio_feats = audio.unsqueeze(1)
        audio_q = self.audio_encoder_q(audio_feats[:, :, 1]) # B x D
        audio_q = nn.functional.normalize(audio_q, dim=1)

        if audio_q.requires_grad:
            self._momentum_update(self.audio_encoder_q, self.audio_encoder_k)
        with torch.no_grad():
            audio_k = self.audio_encoder_k(audio_feats[:, :, 0]) # B x D
            audio_k = nn.functional.normalize(audio_k, dim=1)
            ref = nn.functional.normalize(ref, dim=-1)

        score_batch = torch.einsum('bd,dk->bk', [audio_q, audio_k.transpose(0, 1)]) # B x B
        score_queue = torch.einsum('bd,dk->bk', [audio_q, self.MoCoQueue.clone().detach()])
        score = torch.cat([score_batch, score_queue], dim=1)

        # mask_queue = indices.unsqueeze(1) == self.IndexQueue.unsqueeze(0)
        # mask_batch = indices.unsqueeze(1) == indices.unsqueeze(0)
    
        mask_batch = torch.eye(B).to(audio.device)
        mask_queue = torch.zeros_like(score_queue).to(audio.device)
        mask = torch.cat([mask_batch, mask_queue], dim=1)
        
        mask = mask.to(torch.long)

        if audio_q.requires_grad: self._enqueue(audio_k, ref, indices)

        if not self.queue_full:
            return score, mask.to(score.device)
        

        score_ref_queue = torch.einsum('bd,dk->bk', 
            [ref, self.RefQueue.clone().detach()])
        score_ref_batch = torch.einsum('bd,dk->bk', 
            [ref, ref.clone().detach().transpose(0, 1)])

        score_ref = torch.cat([score_ref_batch, score_ref_queue], dim=1)

        tmp = score_ref.clone().detach()
        tmp[mask.to(torch.bool)] = - np.inf
        _, pos_idx = torch.topk(tmp, self.topK, dim=1)
        mask.scatter_(1, pos_idx, 1)
        score_ref[mask.to(bool)] = -1
        score_ref = score_ref * (-1)
        
        return score, [mask.to(score.device), score_ref.to(score.device)]

    def extract_embedding(self, feats):
        feats = feats.unsqueeze(1)
        emb = self.audio_encoder_q.extract_rep(feats)
        return emb


class MoCoSSL(nn.Module):
    def __init__(self, dim=128, cap_Q=2048, topK=5, momentum=0.9):
        """
        TODO: mask out the same audio
        """
        super(MoCoSSL, self).__init__()

        self.register_buffer('MoCoQueue', torch.randn(dim, cap_Q))
        self.MoCoQueue = nn.functional.normalize(self.MoCoQueue, dim=0)
        self.register_buffer('QueuePtr', torch.zeros(1, dtype=torch.long))

        self.register_buffer('IndexQueue', torch.ones(cap_Q) * -1)
        
        self.queue_full = False
        self.topK = topK
        self.m = momentum

        self.ifhard = False
        
        self.encoder_q = efficientnet(dim)
        self.encoder_k = efficientnet(dim)
        for param_k, param_q in zip(
                self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


    @torch.no_grad()
    def _enqueue(self, keys, indices):
        B = keys.shape[0]
        ptr = self.QueuePtr[0]
        assert self.MoCoQueue.shape[1] % B == 0
        self.MoCoQueue[:, ptr: ptr+B] = keys.T
        self.IndexQueue[ptr: ptr+B] = indices

        ptr = (ptr + B) % self.MoCoQueue.shape[1]
        self.QueuePtr[0] = ptr
        if not ptr: self.queue_full = True

    @torch.no_grad()
    def _momentum_update(self):
        for param_k, param_q in zip(
                self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    def EasyLearing(self, feats, indices):

        feats = feats.unsqueeze(1).repeat(1,3,1,1,1)
        q = self.encoder_q(feats[:, :, 1]) # B x D
        q = nn.functional.normalize(q, dim=1)

        if q.requires_grad:
            self._momentum_update()

        with torch.no_grad():
            k = self.encoder_k(feats[:, :, 0]) # B x D
            k = nn.functional.normalize(k, dim=1)
        s = torch.einsum('bd,dk->bk', [q, k.transpose(0, 1)])
        mask = indices.unsqueeze(1) == indices.unsqueeze(0)
        # mask = torch.eye(s.shape[0]).to(feats.device)
        if q.requires_grad: self._enqueue(k, indices)



        # return s, mask.to(torch.long)
        return s, mask.to(torch.float)


    def HardLearing(self, feats, indices):
        feats = feats.unsqueeze(1).repeat(1,3,1,1,1)
        q = self.encoder_q(feats[:, :, 1]) # B x D
        q = nn.functional.normalize(q, dim=1)

        if q.requires_grad:
            self._momentum_update()

        with torch.no_grad():
            k = self.encoder_k(feats[:, :, 0]) # B x D
            k = nn.functional.normalize(k, dim=1)

        # score_pos = torch.einsum('bd,bd->b', [q, k]).unsqueeze(-1) # B x 1
        score_batch = torch.einsum('bd,dk->bk', [q, k.transpose(0, 1)]) # B x B
        # score_queue = torch.einsum('bd,dk->bk', [q, self.MoCoQueue]) # B x Q
        score_queue = torch.einsum('bd,dk->bk', [q, self.MoCoQueue.clone().detach()]) # B x Q

        # the first B elements in ref_indices are just indices
        mask_idx = indices.unsqueeze(1) == self.IndexQueue.unsqueeze(0)
        # mask_idx = torch.zeros_like(score_queue)
        mask_queue = mask_idx.to(torch.long)


        score = torch.cat([score_batch, score_queue], dim=1)
        mask_batch = indices.unsqueeze(1) == indices.unsqueeze(0)
        mask = torch.cat([mask_batch, mask_queue],dim=1) # B x (B + Q)

        if q.requires_grad: self._enqueue(k, indices)

        
        return score, mask.to(score.device)

    def forward(self, feats, ref_feats, indices):
        if self.ifhard:
            return self.HardLearing(feats, indices)
        return self.EasyLearing(feats, indices)

    def extract_embedding(self, feats):
        feats = feats.unsqueeze(1).repeat(1,3,1,1)
        emb = self.encoder_q.extract_rep(feats)
        return emb
        # return nn.functional.normalize(emb, dim=1)




# class CoSSL(nn.Module):
#     def __init__(self, audio_dim=128, ref_dim=1280, out_dim=256, cap_Q=2048, topK=5, momentum=0.9):

#         super(CoSSL, self).__init__()

#         self.register_buffer('MoCoQueue', torch.randn(audio_dim, cap_Q))
#         # self.MoCoQueue = nn.functional.normalize(self.MoCoQueue, dim=0)
#         self.register_buffer('QueuePtr', torch.zeros(1, dtype=torch.long))

#         self.register_buffer('RefQueue', torch.randn(ref_dim, cap_Q))
#         # self.RefQueue = nn.functional.normalize(self.RefQueue, dim=0)
#         self.register_buffer('IndexQueue', torch.ones(cap_Q) * -1)
        
#         self.queue_full = False
#         self.topK = topK
#         self.m = momentum
#         self.cap_Q = cap_Q

#         self.ifhard = False
        
#         self.audio_encoder_q = Encoder(out_dim)
#         self.audio_encoder_k = Encoder(out_dim)
#         # self.audio_encoder_q = efficientnet()
#         # self.audio_encoder_k = efficientnet()

#         # self.emo_encoder_q = nn.Sequential(
#         #     MLP(in_dim=emo_dim, out_dim=256, n_hidden=3),
#         #     MLP(in_dim=256, out_dim=256, n_hidden=2))
#         # self.emo_encoder_k = nn.Sequential(
#         #     MLP(in_dim=emo_dim, out_dim=256, n_hidden=3),
#         #     MLP(in_dim=256, out_dim=256, n_hidden=2))
#         self._set_key_encoder(self.audio_encoder_q, self.audio_encoder_k)
#         # self._set_key_encoder(self.emo_encoder_q, self.emo_encoder_k)

#     @torch.no_grad()
#     def _set_key_encoder(self, encoder_q, encoder_k):
#         for param_q, param_k in zip(
#                 encoder_q.parameters(), encoder_k.parameters()):
#             param_k.data.copy_(param_q.data)
#             param_k.requires_grad = False

#     @torch.no_grad()
#     def _momentum_update(self, encoder_q, encoder_k):
#         for param_q, param_k in zip(
#                 encoder_q.parameters(), encoder_k.parameters()):
#             param_k.data = param_q.data * (1 - self.m) + param_k.data * self.m

#     @torch.no_grad()
#     def _enqueue(self, keys, ref_keys, indices):
#         B = keys.shape[0]
#         ptr = self.QueuePtr[0]
#         assert self.MoCoQueue.shape[1] % B == 0
#         self.MoCoQueue[:, ptr: ptr+B] = keys.T
#         self.RefQueue[:, ptr: ptr+B] = ref_keys.T
#         self.IndexQueue[ptr: ptr+B] = indices

#         ptr = (ptr + B) % self.MoCoQueue.shape[1]
#         self.QueuePtr[0] = ptr
#         if not ptr: self.queue_full = True

#     def EasyLearing(self, audio, ref, indices):
#         # audio_feats = audio.unsqueeze(1).repeat(1,3,1,1,1)
#         audio_feats = audio.unsqueeze(1)
#         audio_q = self.audio_encoder_q(audio_feats[:, :, 1]) # B x D
#         audio_q = nn.functional.normalize(audio_q, dim=1)

#         if audio_q.requires_grad:
#             self._momentum_update(self.audio_encoder_q, self.audio_encoder_k)
#         with torch.no_grad():
#             audio_k = self.audio_encoder_k(audio_feats[:, :, 0]) # B x D
#             audio_k = nn.functional.normalize(audio_k, dim=1)
#         s = torch.einsum('bd,dk->bk', [audio_q, audio_k.transpose(0, 1)])

#         # mask = torch.eye(s.shape[0]).to(audio.device)
#         mask = indices.unsqueeze(1) == indices.unsqueeze(0)

#         if audio_q.requires_grad: self._enqueue(audio_k, ref, indices)

#         return s, mask.to(torch.long)

#     def HardLearing(self, audio, ref, indices):
#         # audio_feats = audio.unsqueeze(1).repeat(1,3,1,1,1)
#         audio_feats = audio.unsqueeze(1)
#         audio_q = self.audio_encoder_q(audio_feats[:, :, 1]) # B x D
#         audio_q = nn.functional.normalize(audio_q, dim=1)

#         if audio_q.requires_grad:
#             self._momentum_update(self.audio_encoder_q, self.audio_encoder_k)
#         with torch.no_grad():
#             audio_k = self.audio_encoder_k(audio_feats[:, :, 0]) # B x D
#             audio_k = nn.functional.normalize(audio_k, dim=1)
#             ref = nn.functional.normalize(ref, dim=1)

#         # score_pos = torch.einsum('bd,bd->b', [q, k]).unsqueeze(-1) # B x 1
#         score_batch = torch.einsum('bd,dk->bk', [audio_q, audio_k.transpose(0, 1)]) # B x B
#         score_queue = torch.einsum('bd,dk->bk', [audio_q, self.MoCoQueue.clone().detach()])
#         score = torch.cat([score_batch, score_queue], dim=1)

#         # mask_queue = indices.unsqueeze(1) == self.IndexQueue.unsqueeze(0)
#         # mask_batch = indices.unsqueeze(1) == indices.unsqueeze(0)
#         mask_batch = torch.eye(score_batch.shape[0]).to(audio.device)
#         mask_queue = torch.zeros_like(score_queue).to(audio.device)
#         mask = torch.cat([mask_batch, mask_queue], dim=1)
#         mask = mask.to(torch.long)

#         if audio_q.requires_grad: self._enqueue(audio_k, ref, indices)
#         if True:
#         # if not self.queue_full:
#             return score, mask.to(score.device)
        
#         score_ref_queue = torch.einsum('bd,dk->bk', 
#             [ref, self.RefQueue.clone().detach()])
#         score_ref_batch = torch.einsum('bd,dk->bk', 
#             [ref, ref.clone().detach().transpose(0, 1)])
#         score_ref = torch.cat([score_ref_batch, score_ref_queue], dim=1)

#         tmp = score_ref.clone().detach()
#         # tmp[mask.to(torch.bool)] = - np.inf
#         # _, idx = torch.topk(-tmp, self.topK, dim=1)
#         # mask.scatter_(1, idx, 1)
#         tmp[mask.to(torch.bool)] = - np.inf
#         _, pos_idx = torch.topk(tmp, self.topK, dim=1)
#         # _, neg_idx = torch.topk(-tmp, self.topK, dim=1)
#         mask.scatter_(1, pos_idx, 1)
#         score_ref[mask.to(bool)] = (-0.1)
#         # mask.scatter_(1, neg_idx, 1)
#         # score_ref.scatter_(1, neg_idx, 0.1)
#         # score_ref.scatter_(1, pos_idx, -1)
#         # score_ref[mask.to(bool)] = (-1)
#         score_ref = score_ref * (-1)

        
#         return score, [mask.to(score.device), score_ref.to(score.device)]
#         # return score, mask.to(score.device)

#     def forward(self, feats, ref_feats, indices):

#         # if self.ifhard:
#         return self.HardLearing(feats, ref_feats, indices)
#         # return self.EasyLearing(feats, ref_feats, indices)

#     def extract_embedding(self, feats):
#         # feats = feats.unsqueeze(1).repeat(1,3,1,1)
#         feats = feats.unsqueeze(1)
#         emb = self.audio_encoder_q.extract_rep(feats)
#         return emb



# class efficientnet(nn.Module):
#     def __init__(self, out_dim=256, pretrain='efficientnet-b0'):
#         super(efficientnet, self).__init__()
#         self.backbone = EfficientNet.from_pretrained(pretrain)
#         self.mlp = MLP(out_dim=out_dim)

#     def forward(self, x):
#         feats = self.backbone.extract_features(x).mean((2,3))
#         return self.mlp(feats)

#     def extract_rep(self, x):
#         feats = self.backbone.extract_features(x).mean((2,3))
#         return feats