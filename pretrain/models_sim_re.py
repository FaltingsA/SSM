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

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class label_smooth_loss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, focal_factor=0.):
        super(label_smooth_loss, self).__init__()
        eps = smoothing / num_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps
        self.focal_factor = focal_factor
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
        
        # loss = torch.sum(-true_dist * pred, dim=1) * ((1 - torch.exp(torch.sum(true_dist * pred, dim=1))) ** self.focal_factor)
        # return loss.mean()
        return torch.sum(-true_dist * pred, dim=1).mean()

class PatchNet(nn.Module):
  def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
               num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
               drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0.,
               use_learnable_pos_emb=False, train_with_ctx=False,
               num_windows=5, patch_shape=(8, 32), use_patch_transformer=False, hierarchical_num_windows=[1,2,4],):
    super().__init__()
    if use_patch_transformer:
      dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
      self.blocks = nn.ModuleList([
          Block(
              dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
              drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
              init_values=init_values)
          for i in range(depth)])
      self.norm =  norm_layer(embed_dim)
      self.apply(self._init_weights)

    self.num_windows = num_windows
    self.patch_shape = patch_shape
    self.use_patch_transformer = use_patch_transformer

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)

  def forward(self, seq_x, return_attn_map=False):
    B, _, C = seq_x.shape # [B, 8*32, C]

    x = seq_x.reshape(B, self.patch_shape[0], self.patch_shape[1], C).permute(0, 3, 1, 2) # [B, 8, 32, C]
    x = F.adaptive_avg_pool2d(x, (1, self.num_windows)).permute(0, 2, 3, 1).squeeze(1) # [B, num_windows, C]

    if self.use_patch_transformer:
      for blk in self.blocks:
        if return_attn_map:
          x, attn_map = blk(x, seq_x, seq_x, return_attn_map=True)
        else:
          x = blk(x, seq_x, seq_x)
      x = self.norm(x)
    if return_attn_map:
      return x, attn_map
    else:
      return x

class PromptEmbedding(nn.Module):

    def __init__(self, direction_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(direction_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)
    
class FlipAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, online_ln=True, online_target=True,
                 embed_dim=1024, depth=24, num_heads=16, direction_size=4, T=0.2, neg_weight=0.02,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, label_smoothing=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        self.T=T
        self.neg_weight=neg_weight
        self.label_smoothing = label_smoothing
        self.prompt_embd = PromptEmbedding(direction_size,decoder_embed_dim)
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
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
 
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True) # decoder to patch
        

        self.norm_pix_loss = norm_pix_loss

        if online_ln and online_target:
            self.student_norm = LayerNorm(decoder_embed_dim)
            for p in self.student_norm.parameters():
                p.requires_grad = False
        else:
            self.student_norm = nn.Identity()

        # project to higt-level space
        self.projection_layer = self._build_mlp(3, decoder_embed_dim, 4096, decoder_embed_dim)
        
        # get the patch features
        self.patch_extractor = PatchNet(
            embed_dim=decoder_embed_dim,
            depth=2,
            num_heads=decoder_num_heads,
            num_windows=4,
            patch_shape=(self.num_h,self.num_w),
            use_patch_transformer=False,)
    
        self.predictor = self._build_mlp(2, decoder_embed_dim, 4096, decoder_embed_dim)

        self.initialize_weights()

        # build momentum branch
        if online_target:
            self.build_momentum_target(img_size, patch_size, in_chans, embed_dim, num_heads, decoder_depth, direction_size,
                                        mlp_ratio, norm_layer, depth, decoder_embed_dim, decoder_num_heads)
    

    def build_momentum_target(self, img_size, patch_size, in_chans, embed_dim, num_heads, decoder_depth, direction_size,
                                mlp_ratio, norm_layer, depth, decoder_embed_dim, decoder_num_heads):
        # --------------------------------------------------------------------------
        # momentum encoder specifics
        self.mm_patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.mm_norm = norm_layer(embed_dim)

        self.mm_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        # load weight
        self.mm_patch_embed.load_state_dict(self.patch_embed.state_dict())
        for p in self.mm_patch_embed.parameters():
            p.requires_grad = False

        self.mm_norm.load_state_dict(self.norm.state_dict())
        for p in self.mm_norm.parameters():
            p.requires_grad = False

        self.mm_blocks.load_state_dict(self.blocks.state_dict())
        for p in self.mm_blocks.parameters():
            p.requires_grad = False
        # --------------------------------------------------------------------------
        # projection & patch extractor
        self.mm_projection_layer = self._build_mlp(3, decoder_embed_dim, 4096, decoder_embed_dim)
        self.mm_projection_layer.load_state_dict(self.projection_layer.state_dict())
        for p in self.mm_projection_layer.parameters():
            p.requires_grad = False

        self.mm_patch_extractor = PatchNet(
            embed_dim=decoder_embed_dim,
            depth=2,
            num_heads=decoder_num_heads,
            num_windows=4,
            patch_shape=(self.num_h,self.num_w),
            use_patch_transformer=False,)

        self.mm_patch_extractor.load_state_dict(self.patch_extractor.state_dict())
        for p in self.mm_patch_extractor.parameters():
            p.requires_grad = False

        # --------------------------------------------------------------------------
        # momentum decoder specifics
        self.mm_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mm_decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.mm_decoder_norm = norm_layer(decoder_embed_dim)
        
        # load weight
        self.mm_decoder_embed.load_state_dict(self.decoder_embed.state_dict())
        for p in self.mm_decoder_embed.parameters():
            p.requires_grad = False
        
        # if decoder_depth > 0:
        self.mm_decoder_blocks.load_state_dict(self.decoder_blocks.state_dict())
        for p in self.mm_decoder_blocks.parameters():
            p.requires_grad = False

        self.mm_decoder_norm.load_state_dict(self.decoder_norm.state_dict())
        for p in self.mm_decoder_norm.parameters():
            p.requires_grad = False
        # ---------------------------------------------------------------------------

        self.teacher_norm = LayerNorm(decoder_embed_dim, elementwise_affine=False)
        for p in self.teacher_norm.parameters():
            p.requires_grad = False

        self.mm_prompt_embd = PromptEmbedding(direction_size,decoder_embed_dim)
        for p in self.mm_prompt_embd.parameters():
            p.requires_grad = False

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
        
        # torch.nn.init.normal_(self.pos_token, std=.02)
        # torch.nn.init.normal_(self.neg_token, std=.02)

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

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True, use_conv=False):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            if use_conv:
                mlp.append(nn.Conv1d(dim1, dim2, 1, bias=False))
            else:
                mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.cuda.amp.autocast(enabled=False)
    def mm_update(self, mm):
        for param_q, param_k in zip(self.patch_embed.parameters(), self.mm_patch_embed.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        for param_q, param_k in zip(self.blocks.parameters(), self.mm_blocks.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        
        if hasattr(self, 'mm_norm'):
            for param_q, param_k in zip(self.norm.parameters(), self.mm_norm.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_projector'):
            for param_q, param_k in zip(self.projector.parameters(), self.mm_projector.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_decoder_embed'):
            for param_q, param_k in zip(self.decoder_embed.parameters(), self.mm_decoder_embed.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        
        if hasattr(self, 'mm_decoder_blocks'):
            for param_q, param_k in zip(self.decoder_blocks.parameters(), self.mm_decoder_blocks.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_decoder_blocks'):
            for param_q, param_k in zip(self.decoder_blocks.parameters(), self.mm_decoder_blocks.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_decoder_norm'):
            for param_q, param_k in zip(self.decoder_norm.parameters(), self.mm_decoder_norm.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_decoder_pred'):
            for param_q, param_k in zip(self.decoder_pred.parameters(), self.mm_decoder_pred.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)

        if hasattr(self, 'mm_prompt_embd'):
            for param_q, param_k in zip(self.prompt_embd.parameters(), self.mm_prompt_embd.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)

        if hasattr(self, 'mm_projection_layer'):
            for param_q, param_k in zip(self.projection_layer.parameters(), self.mm_projection_layer.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)

        if hasattr(self, 'mm_patch_extractor'):
            for param_q, param_k in zip(self.patch_extractor.parameters(), self.mm_patch_extractor.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)


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

    @torch.no_grad()
    def forward_mm_encoder(self, x):
        # embed patches
        x = self.mm_patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.mm_blocks:
            x = blk(x)

        x = self.mm_norm(x)

        return x

    def forward_decoder(self, x, prompt):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed
         
        direction_embd  = self.prompt_embd(prompt).unsqueeze(1) # [batch,1,dec_dim]
        x = torch.cat((direction_embd, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # remove direction token
        x = x[:,1:,:]
        
        # # predictor projection
        # x = self.decoder_pred(x)
        
        return x

    @torch.no_grad()
    def forward_mm_decoder(self, x, prompt):
        # embed tokens
        x = self.mm_decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed
         
        direction_embd  = self.mm_prompt_embd(prompt).unsqueeze(1) # [batch,1,dec_dim]
        x = torch.cat((direction_embd, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.mm_decoder_blocks:
            x = blk(x)
        x = self.mm_decoder_norm(x)
        
        # remove direction token
        x = x[:,1:,:]
    
        return x

    def forward_patch_proj(self, x):
        # patches_ori = self.patch_extractor(ori_enc_o)
        # patches_fli = self.patch_extractor(fli_enc_o)

        # b, l, c = patches_ori.shape
        # patches_ori = patches_ori.reshape(b*l, c)
        # b, l, c = patches_fli.shape
        # patches_fli = patches_fli.reshape(b*l, c)


        # q_ori = self.encoder_projection_layer(patches_ori)
        # q_ori = self.predictor(q_ori)
        # q_ori = q_ori.reshape(b, l, -1)

        # q_fli = self.encoder_projection_layer(patches_fli)
        # q_fli = self.predictor(q_fli)
        # q_fli = q_fli.reshape(b, l, -1)

        # q_ori = q_ori.view(-1, q_ori.size(-1))
        # q_fli = q_fli.view(-1, q_fli.size(-1))
        x = self.patch_extractor(x)
        b, l, c = x.shape
        x = x.reshape(b*l, c)
        x = self.projection_layer(x)
        x = self.predictor(x)
        x = x.reshape(b, l, -1)
        return x
    
    @torch.no_grad()
    def forward_mm_patch_proj(self, x):
        x = self.mm_patch_extractor(x)
        b, l, c = x.shape
        x = x.reshape(b*l, c)
        x = self.mm_projection_layer(x)
        x = x.reshape(b, l, -1)
        return x

    def compute_contrastive_loss(self, q, k, return_acc=False, temp=1., hard_k=None):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        # Return loss 
        return label_smooth_loss(logits.shape[-1], self.label_smoothing)(logits, labels) * (2 * self.T)

    def compute_pix_loss(self, imgs, pred):
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

    def compute_unigrad_loss(self, pred, target):
        pred = self.student_norm(pred)
        with torch.no_grad():
            target = self.teacher_norm(target)
        
        dense_pred = pred.reshape(-1, pred.shape[-1])
        dense_target = target.reshape(-1, target.shape[-1])

        # compute pos term
        pos_term = ((dense_pred - dense_target)**2).sum(-1).mean()

        # compute neg term
        correlation = (dense_target.T @ dense_target) / dense_target.shape[0]
        torch.distributed.all_reduce(correlation)
        correlation = correlation / torch.distributed.get_world_size()
        
        neg_term = torch.diagonal(dense_pred @ correlation @ dense_pred.T).mean()

        loss = (pos_term + self.neg_weight * neg_term) / pred.shape[-1]

        return loss

    def compute_unigrad_loss_v2(self, pred, target):
        pred = self.student_norm(pred)
        with torch.no_grad():
            target = self.teacher_norm(target)
        
        dense_pred = pred.reshape(-1, pred.shape[-1])
        dense_target = target.reshape(-1, target.shape[-1])

        # compute pos term
        pos_term = ((dense_pred - dense_target)**2).sum(-1).mean()

        # # compute neg term
        # correlation = (dense_target.T @ dense_target) / dense_target.shape[0]
        # torch.distributed.all_reduce(correlation)
        # correlation = correlation / torch.distributed.get_world_size()
        
        # neg_term = torch.diagonal(dense_pred @ correlation @ dense_pred.T).mean()

        # loss = (pos_term + self.neg_weight * neg_term) / pred.shape[-1]
        loss = pos_term  / pred.shape[-1]

        return loss

    def forward(self, imgs, imgs_gt, teacher_mix, pos_prompt, neg_prompt, mm,update_mm, loss_feat_ratio):
        
        #samples, samples_origin, mask_ratio=args.mask_ratio
        latent = self.forward_encoder(imgs)
        pred_pos_feat = self.forward_decoder(latent, pos_prompt)  # [N, L, p*p*3]
        pred_neg_feat = self.forward_decoder(latent, neg_prompt)

        # pixel level
        pred_pos = self.decoder_pred(pred_pos_feat)
        pred_neg = self.decoder_pred(pred_neg_feat)
        loss_pos = self.compute_pix_loss(imgs_gt['pos'], pred_pos)
        loss_neg = self.compute_pix_loss(imgs_gt['neg'], pred_neg)

        loss_pix = loss_pos + loss_neg
        # feature level & EMA branch
        # ada pooling
        # b,l,c
        pred_pos_feat = self.forward_patch_proj(pred_pos_feat)
        pred_neg_feat = self.forward_patch_proj(pred_neg_feat)


        # forward target encoder
        with torch.no_grad():
            if update_mm:
                self.mm_update(mm)
            
            target_latent = self.forward_mm_encoder(teacher_mix)
            target_pos_feat = self.forward_mm_decoder(target_latent, pos_prompt)  # [N, L, p*p*3]
            target_neg_feat = self.forward_mm_decoder(target_latent, neg_prompt)
            
            target_pos_feat = self.forward_mm_patch_proj(target_pos_feat)
            target_neg_feat = self.forward_mm_patch_proj(target_neg_feat)

        # with torch.cuda.amp.autocast(enabled=False):
        #     loss_feat_pos = self.compute_unigrad_loss_v2(pred_pos_feat.float(), target_pos_feat.float())
        #     loss_feat_neg = self.compute_unigrad_loss_v2(pred_neg_feat.float(), target_neg_feat.float())

        # loss_feat = loss_feat_pos + loss_feat_neg

        # loss = 1.0 * loss_pix + 0 * loss_feat

        loss_outputs = {}
        loss_outputs['loss_pix'] = loss_pix.item()
        # loss_outputs['loss_feat'] = loss_feat.item()
        #==================================================
        pred_pos_feat = pred_pos_feat.reshape(-1, pred_pos_feat.shape[-1])
        pred_neg_feat = pred_neg_feat.reshape(-1, pred_neg_feat.shape[-1])
        target_pos_feat = target_pos_feat.reshape(-1, target_pos_feat.shape[-1])
        target_neg_feat = target_neg_feat.reshape(-1, target_neg_feat.shape[-1])

        loss_feat_pos_cl = self.compute_contrastive_loss(pred_pos_feat, target_pos_feat, temp=self.T)
        loss_feat_neg_cl = self.compute_contrastive_loss(pred_neg_feat, target_neg_feat, temp=self.T)
        
        loss_feat_cl = loss_feat_pos_cl + loss_feat_neg_cl
    
        loss = loss + loss_feat_ratio * loss_feat_cl

        loss_outputs['loss_feat_ratio'] = loss_feat_ratio
        loss_outputs['loss_feat_cl'] = loss_feat_cl.item()
        #========================================================
        loss_outputs['loss'] = loss.item()
        return loss, loss_outputs, pred_pos, pred_neg



def flipae_vit_tiny_dec128d4b(**kwargs):
    model = FlipAutoencoderViT(
        embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def flipae_vit_small_dec256d4b(**kwargs):
    model = FlipAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def flipae_sim_vit_small_dec256d4b(**kwargs):
    model = FlipAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6, online_ln=True, online_target=True,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def flipae_vit_small_dec256d1b(**kwargs):
    model = FlipAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=1, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def flipae_vit_small_dec256d2b(**kwargs):
    model = FlipAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=2, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def flipae_vit_small_dec256d6b(**kwargs):
    model = FlipAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=6, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def flipae_vit_small_dec256d8b(**kwargs):
    model = FlipAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def flipae_vit_small_dec256d8b(**kwargs):
    model = FlipAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def flipae_vit_small_dec384d4b(**kwargs):
    model = FlipAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



# set recommended archs

flipae_vit_tiny_str = flipae_vit_tiny_dec128d4b
flipae_vit_small_str = flipae_vit_small_dec256d4b  # decoder: 256 dim, 4 blocks
flipae_sim_vit_small_str = flipae_sim_vit_small_dec256d4b 
flipae_vit_small_strdim384 = flipae_vit_small_dec384d4b  # decoder: 256 dim, 4 blocks
flipae_vit_small_strdec1 = flipae_vit_small_dec256d1b  # decoder: 256 dim, 1 blocks
flipae_vit_small_strdec2 = flipae_vit_small_dec256d2b  # decoder: 256 dim, 2 blocks
flipae_vit_small_strdec6 = flipae_vit_small_dec256d6b  # decoder: 256 dim, 6 blocks
flipae_vit_small_strdec8 = flipae_vit_small_dec256d8b  # decoder: 256 dim, 6 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
