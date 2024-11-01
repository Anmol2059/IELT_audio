import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

"""
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        # self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None
    ):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding

        # seq = torch.arange(n, device = device)
        # dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        # dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        # rel_pos_emb = self.rel_pos_emb(dist).to(q)
        # pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        # dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out), attn
"""

class Attention(nn.Module):
    # Head Token attention: https://arxiv.org/pdf/2210.05958.pdf
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0., proj_drop=0.):
        super().__init__()
        self.num_heads = heads
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.act = nn.GELU()
        self.ht_proj = nn.Linear(dim_head, dim,bias=True)
        self.ht_norm = nn.LayerNorm(dim_head)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_heads, dim))
    
    def forward(self, x, mask=None):
        B, N, C = x.shape

        # head token
        head_pos = self.pos_embed.expand(x.shape[0], -1, -1)
        x_ = x.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        x_ = x_.mean(dim=2)  # now the shape is [B, h, 1, d//h]
        x_ = self.ht_proj(x_).reshape(B, -1, self.num_heads, C // self.num_heads)
        x_ = self.act(self.ht_norm(x_)).flatten(2)
        x_ = x_ + head_pos
        x = torch.cat([x, x_], dim=1)
        
        # normal mhsa
        qkv = self.qkv(x).reshape(B, N+self.num_heads, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N+self.num_heads, C)
        x = self.proj(x)
        
        # merge head tokens into cls token
        cls, patch, ht = torch.split(x, [1, N-1, self.num_heads], dim=1)
        cls = cls + torch.mean(ht, dim=1, keepdim=True) + torch.mean(patch, dim=1, keepdim=True)
        x = torch.cat([cls, patch], dim=1)

        x = self.proj_drop(x)

        return x, attn[:, :, :N, :N]


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadVoting(nn.Module):
        def __init__(self, num_heads, vote_perhead=8, fix=True):
                super(MultiHeadVoting, self).__init__()
                self.fix = fix
                self.num_heads = num_heads
                self.vote_perhead = vote_perhead

                if self.fix:
                        self.kernel = torch.tensor([1, 2, 1], device='cuda').unsqueeze(0).unsqueeze(0).half()
                        self.conv = F.conv1d
                else:
                        self.conv = nn.Conv1d(1, 1, 3, 1, 1)

        def forward(self, x, select_num=None, last=False):
            B, patch_num = x.shape[0], x.shape[3] - 1
            select_num = self.vote_perhead if select_num is None else select_num
            count = torch.zeros((B, patch_num), dtype=torch.int, device='cuda').half()
            score = x[:, :, 0, 1:]
            _, select = torch.topk(score, self.vote_perhead, dim=-1)
            select = select.reshape(B, -1)

            for i, b in enumerate(select):
                count[i, :] += torch.bincount(b, minlength=patch_num).to(count.device)
            if not last:
                count = self.enhace_local(count)
                pass

            patch_value, patch_idx = torch.sort(count, dim=-1, descending=True)
            patch_idx += 1
            return patch_idx[:, :select_num], count

        def enhace_local(self, count):
            B, T = count.shape[0], math.ceil(math.sqrt(count.shape[1]))
            if self.fix:
                count = self.conv(count.unsqueeze(1), self.kernel, stride=1, padding=1).reshape(B, -1)
            else:
                count = self.conv(count.unsqueeze(1)).reshape(B, -1)
            return count

class CrossLayerRefinement(nn.Module):
        def __init__(self, hidden_size, clr_layer):
                super(CrossLayerRefinement, self).__init__()
                self.clr_layer = clr_layer
                self.clr_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        def forward(self, x, cls):
                out = [torch.stack(token) for token in x]
                out = torch.stack(out).squeeze(1)
                out = torch.cat((cls, out), dim=1)
                out, weights = self.clr_layer(out)
                out = self.clr_norm(out)
                return out, weights

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        attn_x, attn_weight = self.attn(x, mask = mask)
        x = attn_x + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x, attn_weight

# Conformer

class IELTEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        cam=True,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        vote_perhead=8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.dim = dim
        self.cam = cam
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal

            ))

        self.clr_layer = ConformerBlock(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            conv_expansion_factor = conv_expansion_factor,
            conv_kernel_size = conv_kernel_size,
            conv_causal = conv_causal
        )  
        if self.cam:
            self.key_layer = ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal
            )

        self.patch_select = MultiHeadVoting(num_heads=heads, vote_perhead=vote_perhead)
        self.clr_encoder = CrossLayerRefinement(dim, self.clr_layer)
        self.count = 0

    def forward(self, x):
        B, N, C = x.shape
        complements = [[] for _ in range(B)]
        for block in self.layers:
            x, attn_weight = block(x)
            select_idx, select_score = self.patch_select(attn_weight, select_num=24)
            for i in range(B):
                selected_token = x[i, select_idx[i, :], :]
                complements[i].extend(selected_token)
        cls_token = x[:, 0].unsqueeze(1)
        clr, weights = self.clr_encoder(complements, cls_token)
        if self.cam:
            sort_idx, _ = self.patch_select(weights, select_num=24, last=True)
            out = []
            for i in range(B):
                out.append(clr[i, sort_idx[i, :]])
            out = torch.stack(out).squeeze(1)
            out = torch.cat((cls_token, out), dim=1)
            key, _ = self.key_layer(out)
            return key[:, 0], clr[:, 0]
        else:
            return clr[:, 0], None
