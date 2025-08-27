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
        return F.avg_pool1d(x, kernel_size=self.window, stride=self.window)

class _FusionHeadGaussian(nn.Module):
    """
    Take concatenated multi-scale features per variable and (optionally) projected future covariates,
    and output Gaussian parameters (mu, log_var) for each horizon step.
    """
    def __init__(self, in_nf: int, target_window: int, hidden: int = 512, dropout: float = 0.1, individual: bool = True):
        super().__init__()
        self.individual = individual
        self.target_window = target_window
        if individual:
            self.nets = nn.ModuleList()
            for _ in range(0):  # placeholder; set properly in reset_nvars
                self.nets.append(None)
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_nf, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 2 * target_window),  # 2*H for mu and log_var
            )
        self._proto = nn.Sequential(
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
            self.nets = nn.ModuleList()
            for _ in range(nvars):
                self.nets.append(self._make_branch().to(device))

    def _make_branch(self):
        # deep copy of prototype
        import copy
        return copy.deepcopy(self._proto)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feats: [B, C, NF]  (per-variable fused features)
        returns: (mu, log_var) each [B, C, H]
        """
        B, C, NF = feats.shape
        if self.individual:
            outs = []
            for i in range(C):
                out = self.nets[i](feats[:, i, :])  # [B, 2H]
                outs.append(out.unsqueeze(1))
            out = torch.cat(outs, dim=1)  # [B, C, 2H]
        else:
            out = self.net(feats)  # [B, C, 2H]
        H2 = out.shape[-1]
        H = H2 // 2
        mu, log_var = out[..., :H], out[..., H:]
        # Stabilize log_var
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var

class EnergyPatchTST(nn.Module):
    """
    Multi-scale + future-covariate fusion + probabilistic head on top of PatchTST encoders.
    This class mirrors the paper "EnergyPatchTST" while reusing the original repo's encoder.
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
        self.patch_len = patch_len
        self.stride = stride
        self.scales = list(scales)
        self.revin = revin
        self.rev_layers = nn.ModuleList([RevIN(c_in, affine=affine, subtract_last=subtract_last) if revin else nn.Identity() for _ in self.scales])
        self.scale_pools = nn.ModuleList([_ScalePool(s) for s in self.scales])

        # per-scale encoder (channel-independent)
        self.encoders = nn.ModuleList()
        self.pos_projs = nn.ModuleList()  # linear from patch_len -> d_model is handled in TSTiEncoder
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.attn_dropout = attn_dropout
        self.dropout = dropout

        self.patch_nums = []
        for s in self.scales:
            Ls = context_window // s
            patch_num = int((Ls - patch_len) / stride + 1)
            self.patch_nums.append(patch_num)
            enc = TSTiEncoder(
                c_in=c_in,
                patch_num=patch_num,
                patch_len=patch_len,
                max_seq_len=1024,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=None,
                d_v=None,
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
            in_nf=fused_nf, target_window=target_window, hidden=max(512, fused_nf // 2), dropout=dropout, individual=individual
        )
        self.individual = individual
        # nvars must be set on first forward since we don't know it at init
        self._head_initialized = False

    def _patchify(self, x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
        # x: [B, C, Ls] -> [B, C, P, N]
        x = x.unfold(dimension=-1, size=patch_len, step=stride)  # [B, C, N, P]
        x = x.permute(0, 1, 3, 2)  # [B, C, P, N]
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

        for sp, rev, enc, s, pn in zip(self.scale_pools, self.rev_layers, self.encoders, self.scales, self.patch_nums):
            xs = sp(x)  # [B, C, Ls]
            if isinstance(rev, RevIN):
                xs = xs.permute(0, 2, 1)
                xs = rev(xs, 'norm')
                xs = xs.permute(0, 2, 1)
            # patchify
            xs = self._patchify(xs, self.patch_len, self.stride)  # [B, C, P, N]
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
    def mc_predict(self, x: torch.Tensor, future_z: Optional[torch.Tensor] = None, mc_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo dropout prediction. Returns (mean, variance) aggregated over mc_samples.
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
            attn_dropout=configs.attn_dropout if hasattr(configs, 'attn_dropout') else 0.0,
            revin=configs.revin if hasattr(configs, 'revin') else True,
            affine=configs.affine if hasattr(configs, 'affine') else True,
            subtract_last=configs.subtract_last if hasattr(configs, 'subtract_last') else False,
            future_dim=self.future_dim,
            future_proj_dim=getattr(configs, 'future_proj_dim', 128),
            individual=configs.individual,
        )
    def forward(self, batch_x, future_z=None):
        return self.model(batch_x, future_z=future_z)
