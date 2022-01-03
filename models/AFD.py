# Attention-based Feature-level Distillation 
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch.nn as nn
import torch
from mmdet.models.utils import build_positional_encoding


class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))


class AFD(nn.Module):
    def __init__(self, qk_dim=256, k=5):
        super(AFD, self).__init__()
        self.attention = Attention(qk_dim=qk_dim, anchor_num=k)

    # self anchor attention
    def forward(self, anchors):
        att_map = self.attention(anchors)
        return att_map

# output이 teacher든 student든 attention map을 뽑는 역활을 하도록 바꿔야 함.
class Attention(nn.Module):
    def __init__(self, qk_dim=256, 
                    anchor_num=5, 
                    positional_encoding=dict(
                    type='SinePositionalEncoding',
                    num_feats=128,
                    normalize=True)):
        super(Attention, self).__init__()
        self.qk_dim = qk_dim
        self.anchor_num = anchor_num

        # build cosine sin positional encoding (from DETR)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # in DETR they using head num 8 (optional)
        self.attention_module = nn.MultiheadAttention(qk_dim, num_heads=16)

    def forward(self, anchors):
        """
        Args:
            anchors = tensor(GT_NUM x K x 256)
        """

        masks = anchors.new_zeros((1, anchors.size(1), anchors.size(2))).to(torch.bool)
        pos_embed = self.positional_encoding(masks)         #positional encoding output = (bs x emb_channel x GT_NUM x k)

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        anchors = anchors.unsqueeze(0).flatten(2).permute(2,0,1)
        #key, query = self.linear_trans_keyquery(anchors)            # GT_K x 1 x 256 (query & key shape same!)
        key, query = anchors.clone(), anchors.clone()
        key = key + pos_embed
        query = query + pos_embed
        att_map = self.attention_module(query, key, key)[1]     # 1 x GT_K x GT_K

        return att_map