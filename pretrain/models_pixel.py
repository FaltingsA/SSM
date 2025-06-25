# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from timm.models.vision_transformer import PatchEmbed, Block
import math
from util.pos_embed import get_2d_sincos_pos_embed
from util.misc import LayerNorm


class UniPixelViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=1024, depth=24, num_heads=16, pixel_type='SR',
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, 
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.num_h = int(img_size[0] / patch_size[0])
        self.num_w = int(img_size[1] / patch_size[1])
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        if pixel_type == 'Seg':
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * 1, bias=True) # decoder to patch
        else:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True) # decoder to patch
        
        self.norm_pix_loss = norm_pix_loss

        self.pixel_type = pixel_type

        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [self.num_h, self.num_w], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], [self.num_h, self.num_w], cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
       
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p, q = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h =  imgs.shape[2] // p
        w =  imgs.shape[3] // q
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w,q))
        x = torch.einsum('nchpwq->nhwpqc', x)
        # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        x = x.reshape(shape=(imgs.shape[0], h * w, p*q * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, p*q *3)
        imgs: (N, 3, H, W)
        """
        p, q = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
        h = self.num_h
        w = self.num_w

        x = x.reshape(shape=(x.shape[0], h, w, p, q, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * q))
        return imgs

    def maskpatchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        p, q = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h =  imgs.shape[2] // p
        w =  imgs.shape[3] // q
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w,q))
        x = torch.einsum('nchpwq->nhwpqc', x)
        # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        x = x.reshape(shape=(imgs.shape[0], h * w, p*q * 1))
        return x

    def maskunpatchify(self, x):
        """
        x: (N, L, p*q *1)
        imgs: (N, 1, H, W)
        """
        p, q = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
        h = self.num_h
        w = self.num_w

        x = x.reshape(shape=(x.shape[0], h, w, p, q, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * q))
        return imgs


    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


    def forward_decoder(self, x):
        # embed tokens
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        return x

    def compute_pix_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if self.pixel_type == 'SR':
            target = self.patchify(imgs)
        elif self.pixel_type == 'Seg':
            target = self.maskpatchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = torch.ones_like(loss)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            
        return loss


    def forward(self, imgs, gt):
        
        #samples, samples_origin, mask_ratio=args.mask_ratio
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.compute_pix_loss(gt, pred)
        
        #========================================================
        loss_outputs = {}
        loss_outputs['loss'] = loss.item()
        return loss, loss_outputs, pred


def pixel_vit_small_dec384d3b(**kwargs):
    model = UniPixelViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=384, decoder_depth=3, decoder_num_heads=2,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




pixel_vit_small = pixel_vit_small_dec384d3b  # decoder: 384 dim, 3 blocks
