import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

# 余弦噪声调度
def cosine_alpha_bar(T, s=0.008, device=None):
    ts = torch.arange(T+1, device=device) / T
    f = torch.cos((ts + s)/(1+s) * math.pi/2) ** 2
    a_bar = f / f[0]
    betas = (1 - (a_bar[1:] / a_bar[:-1]).clamp(min=1e-6)).clamp(max=0.999)
    return betas


# FiLM/AdaLN调制
class FullAttention(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        H = hidden_dim or max(input_dim, cond_dim)
        self.ln_x = nn.LayerNorm(input_dim)
        self.ln_c = nn.LayerNorm(cond_dim)
        self.to_gb = nn.Sequential(
            nn.Linear(cond_dim, H), nn.SiLU(),
            nn.Linear(H, 2 * input_dim)
        )
        nn.init.zeros_(self.to_gb[-1].weight)
        nn.init.zeros_(self.to_gb[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        h = self.ln_x(x)
        c = self.ln_c(cond)
        gamma, beta = self.to_gb(c).chunk(2, dim=-1)
        mod = (1.0 + gamma) * h + beta
        return x + self.gate * self.dropout(mod)


# 零初始化线性层
class ZeroLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


# ControlNet
class ControlNetUNetStyle(nn.Module):
    def __init__(self, num_units: int, control_dim: int, num_levels: int = 3):
        super().__init__()
        self.num_units = num_units
        self.in_proj = nn.Linear(control_dim, num_units)
        self.ctrl_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_units, num_units), nn.SiLU(),
                nn.Linear(num_units, num_units), nn.SiLU()
            ) for _ in range(num_levels)
        ])
        self.t_proj = nn.ModuleList([nn.Linear(num_units, num_units) for _ in range(num_levels)])
        self.zero_inject = nn.ModuleList([ZeroLinear(num_units, num_units) for _ in range(num_levels)])

    def forward(self, t_embeds, control: Tensor):
        if control is None:
            return [None, None, None]
        u = self.in_proj(control)
        outs = []
        for i, t_e in enumerate(t_embeds):
            h = u + self.t_proj[i](t_e)
            h = self.ctrl_blocks[i](h)
            r = self.zero_inject[i](h)
            outs.append(r)
        return outs


# 主去噪网络
class Diffusion(nn.Module):
    def __init__(self, n_steps, latent_dim, num_units, cond_dim=None, dropout=0.0):
        super().__init__()
        self.num_units = num_units
        self.cond_dim = cond_dim or latent_dim
        self.cross_attention = FullAttention(input_dim=num_units, cond_dim=num_units, dropout=dropout)
        self.cond_proj = nn.Linear(self.cond_dim, num_units)
        self.linears = nn.ModuleList([
            nn.Linear(latent_dim, num_units), nn.SiLU(),
            nn.Linear(num_units, num_units), nn.SiLU(),
            nn.Linear(num_units, num_units), nn.SiLU(),
            nn.Linear(num_units, latent_dim),
        ])
        self.step_embeddings = nn.ModuleList([nn.Embedding(n_steps, num_units) for _ in range(3)])
        self.controlnet = ControlNetUNetStyle(num_units=num_units, control_dim=self.cond_dim, num_levels=3)

    def forward(self, x, t, cond, control=None):
        cond_u = self.cond_proj(cond)
        t_embeds = [emb(t) for emb in self.step_embeddings]
        ctrl_res = self.controlnet(t_embeds, control)

        for idx in range(3):
            x = self.linears[2 * idx](x)
            if ctrl_res[idx] is not None:
                x = x + ctrl_res[idx]
            fused_cond = cond_u + t_embeds[idx]
            x = self.cross_attention(x, fused_cond)
            x = self.linears[2 * idx + 1](x)

        return self.linears[-1](x)


# LDM
class LDM(nn.Module):
    def __init__(self, device, latent_dim, timesteps, beta_start=None, beta_end=None,
                 cond_dim=None, predict='v'):
        super().__init__()
        self.timesteps = timesteps
        self.device = device
        self.predict = predict
        betas = cosine_alpha_bar(timesteps, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.diffusion = Diffusion(timesteps, latent_dim, latent_dim, cond_dim=cond_dim or latent_dim, dropout=0.0)

    def addnoise(self, z, t):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1)
        eps = torch.randn_like(z)
        noisy_z = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * z
        return noisy_z, eps

    def denoise(self, z, t, cond, control=None):
        return self.diffusion(z, t, cond, control=control)

    def loss(self, x0, cond, p_uncond=0.1, control=None):
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device, dtype=torch.long)
        x_t, eps = self.addnoise(x0, t)
        mask = (torch.rand(B, device=x0.device) < p_uncond).float().unsqueeze(-1)
        cond_in = cond * (1 - mask)
        ctrl_in = None if (control is None) else control * (1 - mask)
        pred = self.denoise(x_t, t, cond_in, control=ctrl_in)

        if self.predict == 'eps':
            target = eps
        else:
            a = self.alpha_bars[t].unsqueeze(-1).sqrt()
            s = (1 - self.alpha_bars[t]).unsqueeze(-1).sqrt()
            target = a * eps - s * x0

        return F.mse_loss(pred, target)

    @torch.no_grad()
    def _predict_model_out(self, x_t, t, cond, guidance=0.0, control=None):
        if guidance <= 1e-6:
            return self.denoise(x_t, t, cond, control=control)
        out_c = self.denoise(x_t, t, cond, control=control)
        ctrl_zero = torch.zeros_like(control) if control is not None else None
        out_u = self.denoise(x_t, t, torch.zeros_like(cond), control=ctrl_zero)
        return out_u + guidance * (out_c - out_u)

    @torch.no_grad()
    def sample_backward_step(self, x_t, t, cond, guidance=0.0, simple_var=True, control=None):
        n = x_t.shape[0]
        t_tensor = torch.full((n,), t, dtype=torch.long, device=x_t.device)
        model_out = self._predict_model_out(x_t, t_tensor, cond, guidance=guidance, control=control)

        if self.predict == 'eps':
            eps_hat = model_out
        else:
            a_bar_t = self.alpha_bars[t]
            eps_hat = a_bar_t.sqrt() * model_out + (1 - a_bar_t).sqrt() * x_t

        if t == 0:
            noise = 0.0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t) * var.sqrt()

        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        mean = (1 / alpha_t.sqrt()) * (x_t - (beta_t / (1 - self.alpha_bars[t]).sqrt()) * eps_hat)
        return mean + noise

    @torch.no_grad()
    def sample(self, shape, cond, guidance=0.0, simple_var=True, control=None):
        z = torch.randn(shape, device=self.device)
        for t in reversed(range(self.timesteps)):
            z = self.sample_backward_step(z, t, cond, guidance=guidance, simple_var=simple_var, control=control)
        return z
