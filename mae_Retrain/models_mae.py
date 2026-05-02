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
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 save_path="", 
                 bottleneck_norm: str = "none",
                 loss_mode: str = "mse"):
        super().__init__()
        # Optional: per-tile instance normalization (disabled by default).
        # This can be useful if you want each DEM tile to have zero-mean/unit-variance
        # before patch embedding, but typically you will use global normalization from the TRAIN split.
        self.loss_mode = loss_mode
        # bottleneck norm
        if bottleneck_norm in (None, "", "none"):
            self.bottleneck_norm = None
        elif bottleneck_norm == "inst1d":
            self.bottleneck_norm = nn.InstanceNorm1d(embed_dim, affine=True, track_running_stats=False)
        else:
            raise ValueError(f"Unknown bottleneck_norm={bottleneck_norm}")
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.in_chans = in_chans
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        # Optional instance normalization on input (per-tile).
        self.initialize_weights()
        self.save_path=save_path
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
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
        """imgs: (N, C, H, W) -> x: (N, L, patch_size**2 * C)"""
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        c = imgs.shape[1]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x
    def unpatchify(self, x):
        """x: (N, L, patch_size**2 * C) -> imgs: (N, C, H, W)"""
        p = self.patch_embed.patch_size[0]
        c = self.in_chans
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        #print(noise)
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    def middle_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.zeros(N, L, device=x.device)
        n = int(np.sqrt(L))
        #print(n)
        center = (n - 1) / 2.0
        for row_i in range(len(noise)):
            row = noise[row_i]
            #print(row)
            # Fill the matrix based on distance from the center with noise
            for i in range(n):
                for j in range(n):
                    # Calculate the distance from the center
                    distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            
                    # Invert the distance to make values smaller closer to the center
                    value = 1 / (1 + distance)
            
                    # Add some random noise
                    noise_value = np.random.uniform(-0.06, 0.06)
                    row[i*n + j] = value + noise_value
            # Normalize values between 0 and 1
            row = (row - row.min()) / (row.max() - row.min())
            for ind in range(len(noise[0])):
                noise[row_i][ind] = row[ind]
        #print(noise)
                
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    
    def _lcc_patch_from_mask(self, lcc_mask, threshold: float = 0.5):
        """
        Convert pixel-space LCC mask [N,1,H,W] to patch-space mask [N,L].
        Any positive LCC pixel inside a patch marks that patch as LCC/masked.
        """
        p = self.patch_embed.patch_size[0]
        m = F.max_pool2d(lcc_mask.float(), kernel_size=p, stride=p)
        return (m > float(threshold)).flatten(1).float()

    def lcc_priority_masking(self, x, mask_ratio, lcc_patch, priority=10.0):
        """
        Fixed-ratio LCC-priority masking. This is kept for compatibility with
        the old Jan-2026 code: LCC patches receive higher masking priority, but
        the final mask ratio is still fixed by mask_ratio.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        len_keep = max(1, min(L, len_keep))

        noise = torch.rand(N, L, device=x.device)
        score = noise + lcc_patch.float() * float(priority)

        ids_shuffle = torch.argsort(score, dim=1)  # small score -> keep
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, lcc_mask=None, lcc_priority=10.0,
                        lcc_mask_mode="none", lcc_patch_threshold=0.5):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        lcc_patch = None
        if lcc_mask is not None and lcc_mask_mode == "priority":
            lcc_patch = self._lcc_patch_from_mask(lcc_mask, threshold=lcc_patch_threshold)
            x, mask, ids_restore = self.lcc_priority_masking(
                x, mask_ratio, lcc_patch=lcc_patch, priority=lcc_priority
            )
        else:
            # Default upstream behavior: per-sample random masking.
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore, lcc_patch

    def forward_lcc_exact(self, imgs, lcc_mask, lcc_patch_threshold=0.5):
        """
        Exact LCC masking for real bathymetry downstream adaptation.

        Important difference from fixed mask_ratio MAE:
          - The patch mask is determined by each tile's LCC mask.
          - Therefore the actual mask ratio can vary from tile to tile.
          - To support variable visible-token counts inside one batch, this
            function processes the encoder/decoder sample-by-sample and then
            concatenates predictions back to [N,L,P].

        This is slower than fixed-ratio masking but is the cleanest way to avoid
        artificially masking extra non-LCC patches just to force a fixed ratio.
        """
        x_all = self.patch_embed(imgs)
        x_all = x_all + self.pos_embed[:, 1:, :]
        lcc_patch = self._lcc_patch_from_mask(lcc_mask, threshold=lcc_patch_threshold)  # [N,L]

        preds = []
        masks = []
        N, L, D = x_all.shape
        for i in range(N):
            xi = x_all[i:i + 1]              # [1,L,D]
            li = lcc_patch[i].bool()         # True = masked/LCC

            ids_keep = torch.nonzero(~li, as_tuple=False).flatten()
            ids_mask = torch.nonzero(li, as_tuple=False).flatten()

            # Extremely defensive fallback: if the whole tile is LCC, keep one
            # token so the encoder has at least one non-cls token.
            if ids_keep.numel() == 0:
                ids_keep = ids_mask[:1]
                ids_mask = ids_mask[1:]
                li = torch.ones(L, device=imgs.device, dtype=torch.bool)
                li[ids_keep] = False

            ids_shuffle = torch.cat([ids_keep, ids_mask], dim=0)  # keep first, mask after
            ids_restore = torch.argsort(ids_shuffle, dim=0).unsqueeze(0)  # [1,L]

            x_masked = torch.gather(
                xi, dim=1,
                index=ids_keep.view(1, -1, 1).repeat(1, 1, D)
            )

            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(1, -1, -1)
            latent = torch.cat((cls_tokens, x_masked), dim=1)

            for blk in self.blocks:
                latent = blk(latent)
            latent = self.norm(latent)
            latent = self._apply_bottleneck_norm(latent)

            pred_i = self.forward_decoder(latent, ids_restore)  # [1,L,P]
            preds.append(pred_i)
            masks.append(li.float().view(1, L))

        pred = torch.cat(preds, dim=0)
        mask = torch.cat(masks, dim=0)
        return pred, mask, lcc_patch

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        return x
    def forward_loss(self, imgs, pred, mask, lcc_patch=None, loss_on_lcc_only=False):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove
        lcc_patch: optional [N,L], 1 is LCC/river patch
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if loss_on_lcc_only and (lcc_patch is not None):
            weight = mask.float() * lcc_patch.float()
            denom = weight.sum()
            if denom > 0:
                return (loss * weight).sum() / denom

        denom = mask.float().sum()
        if denom > 0:
            return (loss * mask.float()).sum() / denom

        # Batch has no masked patches, e.g. all samples have empty LCC masks in
        # exact mode. Return a differentiable zero rather than NaN.
        return loss.mean() * 0.0

    def _apply_bottleneck_norm(self, latent: torch.Tensor) -> torch.Tensor:
        if getattr(self, "bottleneck_norm", None) is None:
            return latent
        cls, tok = latent[:, :1, :], latent[:, 1:, :]   # [N,1,D], [N,L,D]
        tok = tok.transpose(1, 2)                       # [N,D,L]
        tok = self.bottleneck_norm(tok)
        tok = tok.transpose(1, 2)                       # [N,L,D]
        return torch.cat([cls, tok], dim=1)

    def forward(self, imgs, mask_ratio=0.75, file_name="", lcc_mask=None,
                loss_on_lcc_only=False, lcc_priority=10.0,
                lcc_mask_mode="none", lcc_patch_threshold=0.5):
        """
        lcc_mask_mode:
          - "none"     : ignore lcc_mask and use random MAE masking
          - "priority" : old behavior, fixed mask_ratio but prioritize LCC patches
          - "exact"    : new real-bathymetry mode, mask exactly LCC patches;
                         actual mask ratio varies by tile
        """
        if lcc_mask is not None and lcc_mask_mode == "exact":
            pred, mask, lcc_patch = self.forward_lcc_exact(
                imgs, lcc_mask=lcc_mask, lcc_patch_threshold=lcc_patch_threshold
            )
        else:
            latent, mask, ids_restore, lcc_patch = self.forward_encoder(
                imgs, mask_ratio,
                lcc_mask=lcc_mask,
                lcc_priority=lcc_priority,
                lcc_mask_mode=lcc_mask_mode,
                lcc_patch_threshold=lcc_patch_threshold,
            )
            latent = self._apply_bottleneck_norm(latent)
            if self.save_path != "" and file_name != "":
                torch.save(latent, f'{self.save_path}/{file_name}_latent.pt')
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*C]

        loss = self.forward_loss(
            imgs, pred, mask,
            lcc_patch=lcc_patch,
            loss_on_lcc_only=loss_on_lcc_only,
        )
        return loss, pred, mask
        
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
