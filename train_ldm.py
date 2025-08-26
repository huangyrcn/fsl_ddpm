import torch
from torch import nn, Tensor
import torch.nn.functional as F



class FullAttention(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=None, dropout=0.1):
        super(FullAttention, self).__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim

        # Projection layers
        self.query_proj = nn.Linear(input_dim, self.hidden_dim)
        self.key_proj = nn.Linear(cond_dim, self.hidden_dim)
        self.value_proj = nn.Linear(cond_dim, self.hidden_dim)

        # Optional output projection
        self.out_proj = nn.Linear(self.hidden_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        """
        x: (N, input_dim) - node features
        cond: (N, cond_dim) - conditional input (e.g., class label embeddings or graph latent)
        """
        # Project to Q, K, V
        Q = self.query_proj(x)  # (N, H)
        K = self.key_proj(cond)  # (N, H)
        V = self.value_proj(cond)  # (N, H)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.T) / (Q.shape[-1] ** 0.5)  # (N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (N, N)
        attn_weights = self.dropout(attn_weights)

        # Attend to values
        attn_output = torch.matmul(attn_weights, V)  # (N, H)
        out = self.out_proj(attn_output)  # (N, input_dim)

        return out


class Diffusion(nn.Module):
    def __init__(self, n_steps, latent_dim, num_units):
        super(Diffusion, self).__init__()
        
        self.cross_attention = FullAttention(input_dim=latent_dim, cond_dim=latent_dim)

        self.linears = nn.ModuleList(
            [
                nn.Linear(latent_dim, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, latent_dim),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x, t, cond):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x = self.cross_attention(x, cond)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x


class LDM(nn.Module):
    def __init__(self, device, latent_dim, timesteps, beta_start, beta_end):
        super().__init__()
        # self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        self.diffusion = Diffusion(timesteps, latent_dim, latent_dim)

    def addnoise(self, z, t):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1)
        eps = torch.randn_like(z)  # N x latent_dim
        noisy_z = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * z
        return noisy_z, eps

    def denoise(self, z, t, cond):
        return self.diffusion(z, t, cond)

    def sample_backward(self, noisy_z, cond, simple_var=True):
        # x = torch.randn(shape).to(self.device)  # 初始噪声
        x = noisy_z
        for t in reversed(range(self.timesteps)):
            x = self.sample_backward_step(x, t, cond, simple_var)
        return x

    def sample_backward_step(self, x_t, t, cond, simple_var=True):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device)  # n

        eps = self.denoise(x_t, t_tensor, cond)

        # 计算噪声
        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        # 去噪公式
        # mean = (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])
        mean = (
            1
            / torch.sqrt(self.alphas[t])
            * (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps)
        )
        x_t = mean + noise  # 加入噪声
        return x_t

    def sample(self, shape, cond):
        """使用DDPM反向生成新样本"""
        z = torch.randn(shape, device=self.device)  # 初始噪声
        for t in reversed(range(self.timesteps)):
            z = self.sample_backward_step(z, t, cond)
        return z
