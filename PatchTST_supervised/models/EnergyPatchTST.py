import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# Reuse components from the original repo
from layers.PatchTST_backbone import TSTiEncoder
from layers.RevIN import RevIN


class _ScalePool(nn.Module):
    """Downsample along the time axis using average pooling with kernel=stride=window."""
    def __init__(self, window: int):
        super().__init__()
        self.window = int(window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        if self.window == 1:
            return x
        # ensure length divisible by window by trimming the end
        L = x.shape[-1]
        trim = L % self.window
        if trim:
            x = x[..., :-trim]
        # avg_pool1d expects [B, C, L]
        return F.avg_pool1d(x, kernel_size=self.window, stride=self.window)


class _FusionHeadGaussian(nn.Module):
    """
    Take concatenated multi-scale features per variable and (optionally) projected future covariates,
    and output Gaussian parameters (mu, log_var) for each horizon step.
    """
    def __init__(
        self,
        in_nf: int,
        target_window: int,
        hidden: int = 512,
        dropout: float = 0.1,
        individual: bool = True
    ):
        super().__init__()
        self.individual = individual
        self.target_window = target_window

        # Shared prototype branch
        self._proto = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_nf, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * target_window),  # 2*H for mu and log_var
        )

        if individual:
            self.nets = nn.ModuleList()  # populated on first forward via reset_nvars
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_nf, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2 * target_window),
            )
        self.nvars = None

    def reset_nvars(self, nvars: int, device=None):
        self.nvars = nvars
        if self.individual:
            import copy
            self.nets = nn.ModuleList([copy.deepcopy(self._proto).to(device) for _ in range(nvars)])

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feats: [B, C, NF]  (per-variable fused features)
        returns: (mu, log_var) each [B, C, H]
        """
        B, C, _ = feats.shape
        if self.individual:
            outs = []
            for i in range(C):
                outs.append(self.nets[i](feats[:, i, :]).unsqueeze(1))  # [B, 1, 2H]
            out = torch.cat(outs, dim=1)  # [B, C, 2H]
        else:
            out = self.net(feats)  # [B, C, 2H]

        H = out.shape[-1] // 2
        mu, log_var = out[..., :H], out[..., H:]
        # Stabilize log_var to avoid inf/NaN in NLL
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var


class EnergyPatchTST(nn.Module):
    """
    Multi-scale + future-covariate fusion + probabilistic head on top of PatchTST encoders.
    - Multi-scale: downsampled branches (e.g., 1, 24, 168) each with its own PatchTST encoder.
    - Future-known variables: optional path fused into the head.
    - Probabilistic head: per-variable Gaussian (mu, log_var) + MC-dropout helper.

    Inputs:
      x: [B, C, L]
      future_z (optional): [B, H, E] known future covariates per horizon step
    Outputs:
      mu, log_var: [B, C, H]
    """
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        scales: List[int] = (1, 24, 168),
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        future_dim: int = 0,   # E ; if 0, no future path
        future_proj_dim: int = 128,
        individual: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.c_in = c_in
        self.context_window = context_window
        self.target_window = target_window
        self.base_patch_len = max(1, int(patch_len))
        self.base_stride = max(1, int(stride))
        self.scales = list(scales)
        self.revin = revin

        # RevIN per-scale (or Identity)
        self.rev_layers = nn.ModuleList([
            RevIN(c_in, affine=affine, subtract_last=subtract_last) if revin else nn.Identity()
            for _ in self.scales
        ])
        # AveragePooling per scale
        self.scale_pools = nn.ModuleList([_ScalePool(s) for s in self.scales])

        # Per-scale encoder (channel-independent) with *per-scale* patch/stride, clamped to length
        self.encoders = nn.ModuleList()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.attn_dropout = attn_dropout
        self.dropout = dropout

        # Compute per-scale lengths and patch/stride safely
        self.scale_patch_lens: List[int] = []
        self.scale_strides: List[int] = []
        self.patch_nums: List[int] = []

        for s in self.scales:
            # after pooling by s (trim + stride s), effective length ~ floor(L / s)
            Ls = max(1, context_window // max(1, s))

            pl_s = min(self.base_patch_len, Ls)
            pl_s = max(1, pl_s)
            st_s = min(self.base_stride, pl_s)
            st_s = max(1, st_s)

            patch_num = max(1, int((Ls - pl_s) // st_s + 1))

            self.scale_patch_lens.append(pl_s)
            self.scale_strides.append(st_s)
            self.patch_nums.append(patch_num)

            enc = TSTiEncoder(
                c_in=c_in,
                patch_num=patch_num,
                patch_len=pl_s,             # per-scale
                max_seq_len=1024,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=None, d_v=None,
                d_ff=d_ff,
                norm='BatchNorm',
                attn_dropout=attn_dropout,
                dropout=dropout,
                act="gelu",
                store_attn=False,
                key_padding_mask='auto',
                padding_var=None,
                attn_mask=None,
                res_attention=True,
                pre_norm=False,
                pe='zeros',
                learn_pe=True,
                verbose=False,
            )
            self.encoders.append(enc)

        # Future variable projection path
        self.future_dim = int(future_dim)
        if self.future_dim > 0:
            self.future_proj = nn.Linear(self.future_dim, future_proj_dim)
            self.future_mlp = nn.Sequential(
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(future_proj_dim, future_proj_dim),
            )
            z_nf = future_proj_dim * target_window
        else:
            z_nf = 0

        # Fused head: concatenate flattened features from all scales (+ future features) -> Gaussian params
        fused_nf = sum([d_model * pn for pn in self.patch_nums]) + z_nf
        self.head = _FusionHeadGaussian(
            in_nf=fused_nf,
            target_window=target_window,
            hidden=max(512, fused_nf // 2),
            dropout=dropout,
            individual=individual
        )
        self.individual = individual
        # nvars must be set on first forward since we don't know it at init
        self._head_initialized = False

    def _patchify(self, x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
        """
        x: [B, C, Ls] -> [B, C, P, N]
        Uses per-scale patch_len/stride (already clamped to valid range).
        """
        # unfold returns [B, C, N, P]
        x = x.unfold(dimension=-1, size=patch_len, step=stride)
        # permute to [B, C, P, N] expected by TSTiEncoder
        x = x.permute(0, 1, 3, 2)
        return x

    def forward(self, x: torch.Tensor, future_z: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, C, L]
        future_z: [B, H, E] (known future variables for each horizon step), optional
        Returns (mu, log_var): each [B, C, H]
        """
        B, C, L = x.shape
        assert C == self.c_in, f"expected {self.c_in} variables, got {C}"
        feats = []  # collect flattened features per scale

        for sp, rev, enc, pl_s, st_s in zip(
            self.scale_pools, self.rev_layers, self.encoders, self.scale_patch_lens, self.scale_strides
        ):
            xs = sp(x)  # [B, C, Ls]
            if isinstance(rev, RevIN):
                xs = xs.permute(0, 2, 1)      # [B, Ls, C]
                xs = rev(xs, 'norm')
                xs = xs.permute(0, 2, 1)      # [B, C, Ls]
            # patchify with per-scale params
            xs = self._patchify(xs, pl_s, st_s)          # [B, C, P, N]
            # encode -> [B, C, d_model, N]
            hs = enc(xs)
            # flatten last two dims -> [B, C, d_model * N]
            hs = hs.flatten(start_dim=-2)
            feats.append(hs)

        # future path
        if self.future_dim > 0 and future_z is not None:
            # project each step, then flatten
            z = self.future_proj(future_z)            # [B, H, D]
            z = self.future_mlp(z)                    # [B, H, D]
            z = z.reshape(B, -1)                      # [B, H*D]
            # broadcast to per-variable
            z = z.unsqueeze(1).expand(B, C, z.shape[-1])  # [B, C, H*D]
            feats.append(z)

        fused = torch.cat(feats, dim=-1)  # [B, C, NF]
        if not self._head_initialized:
            self.head.reset_nvars(C, device=fused.device)
            self._head_initialized = True
        mu, log_var = self.head(fused)  # [B, C, H] each
        return mu, log_var

    @torch.no_grad()
    def mc_predict(
        self,
        x: torch.Tensor,
        future_z: Optional[torch.Tensor] = None,
        mc_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo dropout prediction. Returns (mean, total_variance) aggregated over mc_samples.
        total_variance = aleatoric + epistemic.
        """
        self.train()  # enable dropout
        mus = []
        vars_ale = []
        for _ in range(mc_samples):
            mu, log_var = self.forward(x, future_z)
            mus.append(mu)
            vars_ale.append(torch.exp(log_var))  # aleatoric variance
        mu_stack = torch.stack(mus, dim=0)                   # [M, B, C, H]
        mean_mu = mu_stack.mean(dim=0)                       # [B, C, H]
        # epistemic variance from mu samples
        var_epi = mu_stack.var(dim=0, unbiased=False)        # [B, C, H]
        var_ale = torch.stack(vars_ale, dim=0).mean(dim=0)   # [B, C, H]
        total_var = var_ale + var_epi
        self.eval()  # restore eval mode
        return mean_mu, total_var


# ---- Thin wrapper to integrate with repo's Exp_Main (expects module.Model) ----
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        c_in = getattr(configs, 'enc_in', getattr(configs, 'c_in', None))
        context_window = configs.seq_len
        target_window = configs.pred_len
        self.future_dim = getattr(configs, 'future_dim', 0)
        scales = getattr(configs, 'scales', (1, 24, 168))

        # store for shape inference
        self.c_in = c_in
        self.context_window = context_window

        self.model = EnergyPatchTST(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            patch_len=configs.patch_len,
            stride=configs.stride,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            n_layers=configs.e_layers,
            d_ff=configs.d_ff,
            scales=list(scales),
            dropout=configs.dropout,
            attn_dropout=getattr(configs, 'attn_dropout', 0.0),
            revin=getattr(configs, 'revin', True),
            affine=getattr(configs, 'affine', True),
            subtract_last=getattr(configs, 'subtract_last', False),
            future_dim=self.future_dim,
            future_proj_dim=getattr(configs, 'future_proj_dim', 128),
            individual=bool(getattr(configs, 'individual', 1)),
        )

    def forward(self, batch_x, future_z=None):
        # Accept either [B, L, C] (repo default) or [B, C, L]
        if batch_x.ndim != 3:
            raise ValueError(f"Unexpected input rank {batch_x.ndim}; expected 3")
        B, D1, D2 = batch_x.shape
        # repo default: [B, L, C]
        if D1 == self.context_window and D2 == self.c_in:
            batch_x = batch_x.permute(0, 2, 1).contiguous()  # -> [B, C, L]
        # already channel-major: [B, C, L]
        elif D1 == self.c_in:
            pass
        else:
            raise ValueError(
                f"Unexpected input shape {tuple(batch_x.shape)}; "
                f"expected [B, L, {self.c_in}] or [B, {self.c_in}, L] with L={self.context_window}"
            )

        return self.model(batch_x, future_z=future_z)
