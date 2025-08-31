import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

# 添加 sklearn 导入
from sklearn.cluster import KMeans


def cosine_alpha_bar(T, s=0.008, device=None):
    ts = torch.arange(T+1, device=device) / T
    f  = torch.cos((ts + s)/(1+s) * math.pi/2) ** 2
    a_bar = f / f[0]
    betas = (1 - (a_bar[1:] / a_bar[:-1]).clamp(min=1e-6)).clamp(max=0.999)
    return betas


class FullAttention(nn.Module):  # 实为 FiLM/AdaLN 调制，不再做批内注意力
    def __init__(self, input_dim, cond_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        H = hidden_dim or max(input_dim, cond_dim)
        self.ln_x = nn.LayerNorm(input_dim)
        self.ln_c = nn.LayerNorm(cond_dim)
        self.to_gb = nn.Sequential(
            nn.Linear(cond_dim, H), nn.SiLU(),
            nn.Linear(H, 2 * input_dim)
        )
        # AdaLN-Zero：从恒等起步
        nn.init.zeros_(self.to_gb[-1].weight)
        nn.init.zeros_(self.to_gb[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        # x:(B,D), cond:(B,C)，每个样本独立调制，不再跨样本
        h = self.ln_x(x)
        c = self.ln_c(cond)
        gamma, beta = self.to_gb(c).chunk(2, dim=-1)
        mod = (1.0 + gamma) * h + beta
        return x + self.gate * self.dropout(mod)


class ControlNet(nn.Module):
    def __init__(self, n_steps, num_units, control_dim):
        super().__init__()
        self.ctrl_proj = nn.Linear(control_dim, num_units)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(num_units, num_units), nn.SiLU(),
                          nn.Linear(num_units, num_units)) for _ in range(3)
        ])
        for blk in self.blocks:
            nn.init.zeros_(blk[-1].weight); nn.init.zeros_(blk[-1].bias)
        self.gates = nn.Parameter(torch.full((3,), 1e-3))  # 关键：非零起点
        self.step_embeddings = nn.ModuleList([nn.Embedding(n_steps, num_units) for _ in range(3)])

    def forward(self, t, control):
        if control is None: return [None, None, None]
        u = self.ctrl_proj(control)
        outs = []
        for i, emb in enumerate(self.step_embeddings):
            r = self.blocks[i](u + emb(t))          # (B, num_units)
            outs.append(self.gates[i] * r)           # 逐层残差
        return outs


class Diffusion(nn.Module):
    def __init__(self, n_steps, latent_dim, num_units, cond_dim=None, dropout=0.0):
        super(Diffusion, self).__init__()
        self.num_units = num_units
        self.cond_dim = cond_dim or latent_dim

        # 关键：FullAttention 的输入/输出维度改成 num_units（与 x 当前通道一致）
        self.cross_attention = FullAttention(input_dim=num_units, cond_dim=num_units, dropout=dropout)

        # 把条件投到 num_units，便于与时间嵌入相加
        self.cond_proj = nn.Linear(self.cond_dim, num_units)
        
        # 添加ControlNet分支
        self.controlnet = ControlNet(n_steps, num_units, control_dim=self.cond_dim)

        # 你的原始前端/后端线性保持不变
        self.linears = nn.ModuleList(
            [
                nn.Linear(latent_dim, num_units), nn.SiLU(),
                nn.Linear(num_units, num_units),  nn.SiLU(),
                nn.Linear(num_units, num_units),  nn.SiLU(),
                nn.Linear(num_units, latent_dim),
            ]
        )
        # 仍然保留你原来的 3 个时间嵌入（每层一个），输出维度= num_units
        self.step_embeddings = nn.ModuleList(
            [nn.Embedding(n_steps, num_units) for _ in range(3)]
        )

    def forward(self, x, t, cond, control=None):
        # 将 cond 先投到 num_units，后与每层的 t-emb 相加，作为 FiLM 的条件
        cond_u = self.cond_proj(cond)  # (B, num_units)
        
        # 获取ControlNet残差
        ctrl_res = self.controlnet(t, control)  # [r1, r2, r3] or None

        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)              # (B, num_units)

            x = self.linears[2 * idx](x)                 # (B, num_units)
            fused_cond = cond_u + t_embedding            # (B, num_units)
            
            # 注入ControlNet残差
            if ctrl_res[idx] is not None:
                fused_cond = fused_cond + ctrl_res[idx]  # ← 注入残差
                
            x = self.cross_attention(x, fused_cond)      # FiLM 调制（不再跨样本）

            x = self.linears[2 * idx + 1](x)             # ReLU

        x = self.linears[-1](x)                          # (B, latent_dim)
        return x


class LDM(nn.Module):
    def __init__(self, device, latent_dim, timesteps, beta_start=None, beta_end=None,
                 cond_dim=None, predict='v'):
        super().__init__()
        self.timesteps = timesteps
        self.device = device
        self.predict = predict  # 'v' or 'eps'

        # 改成 cosine 日程，并注册为 buffer
        betas = cosine_alpha_bar(timesteps, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

        # 传 cond_dim，默认与 latent_dim 相同
        self.diffusion = Diffusion(timesteps, latent_dim, latent_dim, cond_dim=cond_dim or latent_dim, dropout=0.0)

    def addnoise(self, z, t):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1)
        eps = torch.randn_like(z)  # N x latent_dim
        noisy_z = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * z
        return noisy_z, eps

    def denoise(self, z, t, cond, control=None):
        return self.diffusion(z, t, cond, control=control)

    # ===== 可选：训练损失（推荐 v-pred + CFG 训练置零条件） =====
    def loss(self, x0, cond, p_uncond=0.1, control=None):
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device, dtype=torch.long)
        x_t, eps = self.addnoise(x0, t)

        # Classifier-Free：以 p_uncond 概率将条件置零
        mask = (torch.rand(B, device=x0.device) < p_uncond).float().unsqueeze(-1)
        cond_in = cond * (1 - mask)
        
        # 同步置零control（保证CFG对称性）
        control_in = None
        if control is not None:
            control_in = control * (1 - mask)

        pred = self.denoise(x_t, t, cond_in, control=control_in)

        if self.predict == 'eps':
            target = eps
        else:  # v = sqrt(a_bar)*eps - sqrt(1-a_bar)*x0
            a = self.alpha_bars[t].unsqueeze(-1).sqrt()
            s = (1 - self.alpha_bars[t]).unsqueeze(-1).sqrt()
            target = a * eps - s * x0

        return F.mse_loss(pred, target)

    def sample_backward(self, noisy_z, cond, guidance=0.0, simple_var=True):
        # x = torch.randn(shape).to(self.device)  # 初始噪声
        x = noisy_z
        for t in reversed(range(self.timesteps)):
            x = self.sample_backward_step(x, t, cond, guidance=guidance, simple_var=simple_var)
        return x

    # ===== 采样：支持 CFG =====
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

        # 将模型输出转为 ε̂
        if self.predict == 'eps':
            eps_hat = model_out
        else:  # v -> eps
            a_bar_t = self.alpha_bars[t]
            eps_hat = a_bar_t.sqrt() * model_out + (1 - a_bar_t).sqrt() * x_t

        # 计算噪声
        if t == 0:
            noise = 0.0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t) * var.sqrt()

        # 去噪公式
        alpha_t = self.alphas[t]; beta_t = self.betas[t]
        mean = (1 / alpha_t.sqrt()) * (x_t - (beta_t / (1 - self.alpha_bars[t]).sqrt()) * eps_hat)
        return mean + noise

    @torch.no_grad()
    def sample(self, shape, cond, guidance=0.0, simple_var=True, control=None):
        """使用DDPM反向生成新样本，支持CFG"""
        z = torch.randn(shape, device=self.device)  # 初始噪声
        for t in reversed(range(self.timesteps)):
            z = self.sample_backward_step(z, t, cond, guidance=guidance, simple_var=simple_var, control=control)
        return z
