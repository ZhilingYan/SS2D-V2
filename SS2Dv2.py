import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class SS2Dv2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True
        
        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()
        
        self.forward_core = self.forward_corev0
        
        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        A_log = repeat(A_log, "nheads -> r nheads", r=4)
        A_log = A_log.flatten(0, 1)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        D = torch.ones(self.nheads, device=device)
        D = repeat(D, "nheads -> r nheads", r=4)
        D = D.flatten(0, 1)
        self.Ds = nn.Parameter(D)
        self.Ds._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


    def forward_corev0(self, x: torch.Tensor, dt: torch.Tensor, seq_idx=None):
        self.mamba_chunk_scan_combined = mamba_chunk_scan_combined
        
        B, C, H, W = x.shape
        L = H * W
        K = 4
        
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        dts = F.softplus(dt + self.dt_bias)  # (B, H, W, nheads)

        xBC_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xBCs = torch.cat([xBC_hwwh, torch.flip(xBC_hwwh, dims=[-1])], dim=1) # (B, K=4, conv_dim, L)

        xs, Bs, Cs = torch.split(xBCs, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=2)

        xs = xs.float().view(B, -1, L).permute(0, 2, 1).contiguous() # (B, L, K * d_inner)
        dts = dts.contiguous().float().view(B, L, -1) # (B, L, nheads)
        Bs = Bs.float().permute(0, 3, 1, 2).contiguous().view(B, L, -1) # (B, L, self.ngroups * self.d_state)
        Cs = Cs.float().permute(0, 3, 1, 2).contiguous().view(B, L, -1) # (B, L, self.ngroups * self.d_state)
        Ds = self.Ds.float().view(-1) # (K * nheads)
        As = -torch.exp(self.A_log).view(-1)  # (K * nheads)
        
        out_y = self.mamba_chunk_scan_combined(
            rearrange(xs, "b l (h p) -> b l h p", p=self.headdim),
            dts,
            As,
            rearrange(Bs, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(Cs, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=Ds,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        out_y = rearrange(out_y, "b l h p -> b l (h p)").view(B, L, K, -1)
        out_y = out_y.permute(0, 2, 3, 1).contiguous() # (B, K, d_inner, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y


    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        zxbcdt = self.in_proj(x)  # (B, H, W, d_in_proj)
        
        z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            ) # z: (B, H, W, d_inner)
        
        xBC = xBC.permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC)) # (B, conv_dim, H, W)
        y1, y2, y3, y4 = self.forward_core(xBC, dt)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4 # (B, d_inner, L)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, H, W, d_inner)
        
        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out