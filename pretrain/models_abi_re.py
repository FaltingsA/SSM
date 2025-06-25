# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import math
from typing import Optional, Callable
from torchvision.models import resnet
from functools import partial
import torch
import torch.nn as nn
from torch.nn import MSELoss
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

class ResTranformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, d_inner=2048, dropout=0.1, activation='relu', backbone_ln=2):
        super().__init__()
        self.resnet = resnet45()
        self.pos_encoder = PositionalEncoding(d_model, max_len=8 * 32)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=d_inner, dropout=dropout, activation=activation)
        self.transformer = TransformerEncoder(encoder_layer, backbone_ln)

    def forward(self, images):
        feature = self.resnet(images)
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).permute(2, 0, 1)
        feature = self.pos_encoder(feature)
        feature = self.transformer(feature)
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        return feature

class FlipAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # abinet encoder specifics
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
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.pos_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.neg_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

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
        # torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)
        
        torch.nn.init.normal_(self.pos_token, std=.02)
        torch.nn.init.normal_(self.neg_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
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

   

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        
        pos_tokens = self.pos_token.repeat(x.shape[0], 1, 1)
        neg_tokens = self.neg_token.repeat(x.shape[0], 1, 1)
        x_pos = torch.cat((pos_tokens, x), dim=1)
        x_neg = torch.cat((neg_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x_pos = blk(x_pos)
        x_pos = self.decoder_norm(x_pos)

        # predictor projection
        x_pos = self.decoder_pred(x_pos)

        # remove direction token
        x_pos = x_pos[:, 1:, :]
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x_neg = blk(x_neg)
        x_neg = self.decoder_norm(x_neg)

        # predictor projection
        x_neg = self.decoder_pred(x_neg)

        # remove direction token
        x_neg = x_neg[:, 1:, :]
        
#         # apply Transformer blocks
#         for blk in self.decoder_blocks:
#             x = blk(x)
#         x = self.decoder_norm(x)

#         # predictor projection
#         x = self.decoder_pred(x)

#         # remove direction token
#         x = x[:, 1:, :]

        return x_pos, x_neg

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = torch.ones_like(loss)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            
        return loss

    def forward(self, imgs, imgs_gt,  mask_ratio=0.75):
        
        #samples, samples_origin, mask_ratio=args.mask_ratio
        latent = self.forward_encoder(imgs)
        pred_pos, pred_neg = self.forward_decoder(latent)  # [N, L, p*p*3]
    
        loss_pos = self.forward_loss(imgs_gt['pos'], pred_pos)
        loss_neg = self.forward_loss(imgs_gt['neg'], pred_neg)
        
        loss = loss_pos * 1.0 + loss_neg * 1.0
        
        return loss, pred_pos, pred_neg



# def flipae_vit_base_patch16_dec256d4b(**kwargs):
#     model = FlipAutoencoderViT(
#         embed_dim=384, depth=12, num_heads=6,
#         decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_large_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_huge_patch14_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


def flipae_vit_small_dec256d4b(**kwargs):
    model = FlipAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
flipae_vit_small_str = flipae_vit_small_dec256d4b  # decoder: 256 dim, 4 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
