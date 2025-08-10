# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .modules import TransformerDecoderLayer, TransformerDecoder, Encoder, TokenEmbedding, VisionTransformerEncoder
# from custom_vit import VisionTransformerEncoder
# from custom_vit_modules import Block, PatchEmbed

class TransOCR(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], query_type:str,
                 enc_embed_dim: int, enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_embed_dim: int, dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool, standard: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        # self.map = True
        self.map = (enc_embed_dim!=dec_embed_dim)
        # self.encoder = Encoder(img_size, patch_size, embed_dim=enc_embed_dim, depth=enc_depth, num_heads=enc_num_heads,
        #                        mlp_ratio=enc_mlp_ratio)
        if standard:
            self.encoder = Encoder(img_size, patch_size, embed_dim=enc_embed_dim, depth=enc_depth, num_heads=enc_num_heads,
                               mlp_ratio=enc_mlp_ratio)
        else:
            self.encoder = VisionTransformerEncoder(
                img_size=img_size, 
                patch_size=patch_size, 
                in_chans=3, 
                num_classes=0, 
                embed_dim=enc_embed_dim, 
                depth=enc_depth,
                num_heads=enc_num_heads, 
                mlp_ratio=enc_mlp_ratio, 
                qkv_bias=True, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                init_values=0.,
                use_learnable_pos_emb=False)
        
        decoder_layer = TransformerDecoderLayer(dec_embed_dim, dec_num_heads, dec_embed_dim * dec_mlp_ratio, dropout)

        if self.map:
            self.dec_embd = nn.Linear(enc_embed_dim, dec_embed_dim)
        
        self.decoder = TransformerDecoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(dec_embed_dim))

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(dec_embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), dec_embed_dim)

        # +1 for <eos>
        if query_type == 'learn':
            self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, dec_embed_dim))

        self.dropout = nn.Dropout(p=dropout)

        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img: torch.Tensor):
        # return self.encoder(img)
        # return self.dec_embd(self.encoder(img))
        if self.map:
            return self.dec_embd(self.encoder(img))
        else:
            return self.encoder(img)

    def decode(self, tgt, memory, tgt_mask=None,  tgt_padding_mask=None):
            
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))

        tgt = self.dropout(tgt_emb)

        return self.decoder(tgt, memory, tgt_mask, tgt_padding_mask)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id

            logits = []
            # print(" tgt_in shape:", tgt_in.shape)
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j])
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out[:,i,:])
                # print('pi shape',p_i.shape)
                logits.append(p_i)
                if j < num_steps:
                    # print(" tgt_in shape:", tgt_in.shape)
                    # print(" tgt_in [j] shape:", tgt_in[:, j].shape)
                    # print("cur j:",j)
                    # print("cur p_i shape:",p_i.shape)
                    # print("cur p_i squeeze shape:",p_i.shape)
                    # print("cur p_i squeeze:",p_i.squeeze())
                    # print("value :", p_i.squeeze().argmax(-1))
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break
            # print('logits 0', logits[0].shape, logits[0])
            logits = torch.stack(logits, dim=1)
            # print('lsp logits', logits.shape)
        return logits

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(images)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()

        num_steps = tgt_in.size(1)
        tgt_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask)
        logits = self.head(out).flatten(end_dim=1)
        loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
        loss_numel += n

        loss /= loss_numel

        self.log('loss', loss)
        return loss
