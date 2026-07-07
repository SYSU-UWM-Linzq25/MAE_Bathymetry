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

    def _valid_patch_from_mask(self, valid_mask):
        """Return 1 only when every pixel in a patch is valid.

        One NoData pixel invalidates the complete patch. Invalid patches are
        removed from encoder input and are not prediction/loss/RMSE targets.
        """
        p = self.patch_embed.patch_size[0]
        valid_fraction = F.avg_pool2d(
            valid_mask.float(), kernel_size=p, stride=p
        )
        return (valid_fraction >= (1.0 - 1.0e-6)).flatten(1).float()

    def _core_patch_mask_like(self, patch_mask, imgs, radius: int = 3):
        """Centered square patch mask used only for optional core loss.

        The encoder still receives every valid known patch across the full
        tile, and all usable river patches across the full tile remain masked
        from the encoder. Only the loss/evaluation mask is restricted.
        """
        p = self.patch_embed.patch_size[0]
        gh = int(imgs.shape[-2] // p)
        gw = int(imgs.shape[-1] // p)
        if gh * gw != patch_mask.shape[1]:
            raise ValueError(
                f"Patch grid mismatch: {gh}x{gw} vs L={patch_mask.shape[1]}"
            )
        r = int(radius)
        if r < 0:
            raise ValueError(f"core_patch_radius must be >= 0, got {r}")
        cy, cx = gh // 2, gw // 2
        y0, y1 = max(0, cy - r), min(gh, cy + r + 1)
        x0, x1 = max(0, cx - r), min(gw, cx + r + 1)
        core = torch.zeros(
            (gh, gw), device=patch_mask.device, dtype=patch_mask.dtype
        )
        core[y0:y1, x0:x1] = 1
        return core.flatten().unsqueeze(0).expand_as(patch_mask)

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

    def forward_lcc_exact(
        self, imgs, lcc_mask, valid_mask=None, lcc_patch_threshold=0.5
    ):
        """Exact river masking with a third ignored/NoData patch state.

        Patch states:
          * visible: valid patch with no river-mask pixel -> encoder input;
          * prediction: valid patch with >=1 river-mask pixel -> decoder target;
          * ignored: patch with >=1 NoData pixel -> neither input nor target.

        Ignored patches are absent from both encoder and decoder attention. The
        returned full-grid prediction tensor contains zeros only as storage
        placeholders at ignored positions; those positions are never used by
        loss, RMSE, reconstruction, or visualization.
        """
        x_all = self.patch_embed(imgs)
        x_all = x_all + self.pos_embed[:, 1:, :]
        lcc_patch = self._lcc_patch_from_mask(
            lcc_mask, threshold=lcc_patch_threshold
        )  # [N,L]

        if valid_mask is None:
            valid_patch = torch.ones_like(lcc_patch)
        else:
            valid_patch = self._valid_patch_from_mask(valid_mask)

        prediction_patch = lcc_patch.bool() & valid_patch.bool()
        visible_patch = (~lcc_patch.bool()) & valid_patch.bool()
        ignored_patch = ~valid_patch.bool()

        preds = []
        masks = []
        N, L, D = x_all.shape
        for i in range(N):
            xi = x_all[i:i + 1]
            ids_keep = torch.nonzero(visible_patch[i], as_tuple=False).flatten()
            ids_valid = torch.nonzero(
                valid_patch[i].bool(), as_tuple=False
            ).flatten()

            if ids_keep.numel() == 0:
                raise RuntimeError(
                    'A tile has zero valid visible patches after NoData removal. '
                    'Increase data quality or filter it with '
                    '--min_valid_visible_patch_ratio.'
                )

            x_masked = torch.gather(
                xi, dim=1,
                index=ids_keep.view(1, -1, 1).repeat(1, 1, D),
            )

            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(1, -1, -1)
            latent = torch.cat((cls_tokens, x_masked), dim=1)

            for blk in self.blocks:
                latent = blk(latent)
            latent = self.norm(latent)
            latent = self._apply_bottleneck_norm(latent)

            # Decode only valid patches. Ignored/NoData patches are absent from
            # decoder self-attention, not merely excluded from the loss.
            pred_i = self.forward_decoder_valid_subset(
                latent, ids_keep=ids_keep, ids_valid=ids_valid, full_length=L
            )
            preds.append(pred_i)
            masks.append(prediction_patch[i].float().view(1, L))

        pred = torch.cat(preds, dim=0)
        mask = torch.cat(masks, dim=0)
        return pred, mask, lcc_patch, valid_patch

    def forward_decoder_valid_subset(
        self, latent, ids_keep, ids_valid, full_length
    ):
        """Decode only valid patches, then scatter to the full patch grid.

        ``ids_keep`` are the visible valid patch indices in encoder-token order.
        ``ids_valid`` are all valid patch indices in spatial order. Invalid
        patches never enter decoder attention.
        """
        x = self.decoder_embed(latent)
        n_valid = int(ids_valid.numel())
        if n_valid <= 0:
            raise RuntimeError('No valid patches available for decoder.')

        # Under AMP/autocast, ``x`` may be float16 while the learnable
        # mask token and positional embeddings remain float32 parameters.
        # Indexed assignment does not perform autocast automatically, so
        # explicitly align these tensors with the decoder activation dtype.
        mask_token = self.mask_token.to(device=x.device, dtype=x.dtype)
        decoder_pos_embed = self.decoder_pos_embed.to(
            device=x.device, dtype=x.dtype
        )

        tokens = mask_token.repeat(1, n_valid, 1)
        local_keep = torch.searchsorted(ids_valid, ids_keep)
        tokens[:, local_keep, :] = x[:, 1:, :]

        cls = x[:, :1, :] + decoder_pos_embed[:, :1, :]
        pos_valid = decoder_pos_embed[:, ids_valid + 1, :]
        tokens = tokens + pos_valid
        x = torch.cat([cls, tokens], dim=1)

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)[:, 1:, :]

        full = x.new_zeros(
            (1, int(full_length), self.patch_embed.patch_size[0] ** 2 * self.in_chans)
        )
        full[:, ids_valid, :] = x
        return full

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
    def forward_loss(
        self, imgs, pred, loss_mask, lcc_patch=None, valid_patch=None,
        loss_on_lcc_only=False, loss_pixel_mask=None,
    ):
        """MSE on selected patches or selected pixels.

        Backward compatible behavior:
          * if loss_pixel_mask is None, compute patch-mean MSE and weight by
            patch-level loss_mask exactly as before.

        MAE v2 dual-mask behavior:
          * if loss_pixel_mask is provided, patchify it to [N,L,P] and compute
            pixel-weighted MSE. This lets the encoder hide full water patches
            while the loss is computed only on the final pixel-level bathy
            target mask.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        err2 = (pred - target) ** 2

        # New pixel-level loss path.
        if loss_pixel_mask is not None:
            pixel_w = self.patchify(loss_pixel_mask.float())  # [N,L,p*p*C]
            pixel_w = (pixel_w > 0.5).float()

            # Restrict to decoder/prediction patches selected by the hidden mask.
            if loss_mask is not None:
                pixel_w = pixel_w * loss_mask.float().unsqueeze(-1)
            if valid_patch is not None:
                pixel_w = pixel_w * valid_patch.float().unsqueeze(-1)
            if loss_on_lcc_only and lcc_patch is not None:
                pixel_w = pixel_w * lcc_patch.float().unsqueeze(-1)

            denom = pixel_w.sum()
            if denom > 0:
                return (err2 * pixel_w).sum() / denom

            # Defensive differentiable zero.
            return err2.mean() * 0.0

        # Original patch-level loss path.
        loss = err2.mean(dim=-1)  # [N,L]

        weight = loss_mask.float()
        if valid_patch is not None:
            weight = weight * valid_patch.float()
        if loss_on_lcc_only and lcc_patch is not None:
            weight = weight * lcc_patch.float()

        denom = weight.sum()
        if denom > 0:
            return (loss * weight).sum() / denom

        # Differentiable zero for a defensive empty batch. Dataset filtering
        # should normally prevent this situation.
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
                valid_mask=None, loss_pixel_mask=None, loss_on_lcc_only=False, lcc_priority=10.0,
                lcc_mask_mode="none", lcc_patch_threshold=0.5,
                loss_region_mode="all", core_patch_radius=3,
                return_aux_masks=False):
        """
        lcc_mask_mode:
          - "none"     : ignore lcc_mask and use random MAE masking
          - "priority" : old behavior, fixed mask_ratio but prioritize LCC patches
          - "exact"    : new real-bathymetry mode, mask exactly LCC patches;
                         actual mask ratio varies by tile
        """
        if lcc_mask is not None and lcc_mask_mode == "exact":
            pred, prediction_mask, lcc_patch, valid_patch = self.forward_lcc_exact(
                imgs, lcc_mask=lcc_mask, valid_mask=valid_mask,
                lcc_patch_threshold=lcc_patch_threshold,
            )
            mode = str(loss_region_mode).lower()
            if mode == "all":
                loss_mask = prediction_mask
            elif mode == "core":
                core_mask = self._core_patch_mask_like(
                    prediction_mask, imgs, radius=core_patch_radius
                )
                loss_mask = prediction_mask * core_mask
            else:
                raise ValueError(
                    f"loss_region_mode must be 'all' or 'core', got {loss_region_mode}"
                )
        else:
            valid_patch = None
            latent, prediction_mask, ids_restore, lcc_patch = self.forward_encoder(
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
            loss_mask = prediction_mask

        loss = self.forward_loss(
            imgs, pred, loss_mask,
            lcc_patch=lcc_patch,
            valid_patch=valid_patch,
            loss_on_lcc_only=loss_on_lcc_only,
            loss_pixel_mask=loss_pixel_mask,
        )
        if return_aux_masks:
            return loss, pred, loss_mask, prediction_mask
        return loss, pred, loss_mask
        
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
