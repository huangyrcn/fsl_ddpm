import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

# =========================
# 已有: 余弦噪声调度 (保留不变)
# =========================
def cosine_alpha_bar(T, s=0.008, device=None):
    ts = torch.arange(T+1, device=device) / T
    f = torch.cos((ts + s)/(1+s) * math.pi/2) ** 2
    a_bar = f / f[0]
    betas = (1 - (a_bar[1:] / a_bar[:-1]).clamp(min=1e-6)).clamp(max=0.999)
    return betas

# =========================
# 已有: ZeroLinear (保留不变)
# =========================
class ZeroLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        nn.init.zeros_(self.weight); nn.init.zeros_(self.bias)

# =========================
# 新增: 时间嵌入（正弦 + 2层 MLP）
# =========================
def sinusoidal_time_emb(t: Tensor, dim: int) -> Tensor:
    # t: [B] (float 或 long)
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / (half - 1)))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
    return emb

class TimeMLP(nn.Module):
    def __init__(self, time_dim=256, out_dim=256):
        super().__init__()
        self.in_dim = time_dim
        self.proj = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, t: Tensor):
        # t: [B], 先做正弦，再 MLP
        t_emb = sinusoidal_time_emb(t, self.in_dim)
        return self.proj(t_emb)

# =========================
# 新增: FiLM 头 (为每层产生 gamma/beta)
# =========================
class FiLMHead(nn.Module):
    def __init__(self, cond_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * out_dim)
        )
        # 最后一层不用置零，FiLM 需要可用初始值
    def forward(self, cond_vec: Tensor):
        gb = self.net(cond_vec)         # [B, 2*out_dim]
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta

# =========================
# 新增: Zero-MLP（零初始化残差控制分支）
# =========================
class ZeroMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = ZeroLinear(hidden, out_dim)  # 末层零初始化
        self.act = nn.SiLU()
        # 可选可学习缩放门
        self.gate = nn.Parameter(torch.zeros(1))
    def forward(self, h: Tensor) -> Tensor:
        return self.gate * self.fc2(self.act(self.fc1(h)))

# =========================
# 新增: 残差 MLP Block (LN→FiLM→MLP→残差 + Zero-MLP)
# =========================
class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, hidden: int = None, use_zero_mlp: bool = True):
        super().__init__()
        H = hidden or dim
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, H)
        self.fc2 = nn.Linear(H, dim)
        self.film = FiLMHead(cond_dim, H, dim)
        self.use_zero = use_zero_mlp
        if use_zero_mlp:
            # Zero-MLP 的输入拼 [h, cond_vec]
            self.zero = ZeroMLP(dim + cond_dim, H, dim)

    def forward(self, h: Tensor, cond_vec: Tensor):
        # FiLM 调制
        gamma, beta = self.film(cond_vec)       # [B,dim], [B,dim]
        y = self.ln(h) * (1 + gamma) + beta
        y = self.fc2(F.silu(self.fc1(y)))
        h = h + y                                # 主残差
        if self.use_zero:
            h = h + self.zero(torch.cat([h, cond_vec], dim=-1))  # 控制残差（零起点）
        return h

# =========================
# 新增: 向量版 U-Net 主干
# =========================
class VectorUNet(nn.Module):
    """
    widths: 例如 (128, 256, 512)
    time_dim/cond_dim: 时间/条件嵌入维度（默认 256）
    out_param: "v" 或 "eps"
    self_condition: 是否支持拼接 x0_sc
    """
    def __init__(
        self,
        x_dim: int = 128,
        widths=(128, 256, 512),
        time_dim: int = 256,
        cond_dim: int = 256,
        n_blocks_per_stage: int = 2,
        use_zero_mlp: bool = True,
        out_param: str = "v",
        self_condition: bool = True,
        cond_in_dim: int = None,   # 原始 cond 输入维度（类原型维度）；若 None 则与 cond_dim 相同
    ):
        super().__init__()
        assert out_param in ("v", "eps")
        self.out_param = out_param
        self.self_condition = self_condition
        self.x_dim = x_dim

        self.time_mlp = TimeMLP(time_dim=time_dim, out_dim=cond_dim)
        self.cond_in = cond_in_dim or cond_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_in, cond_dim, bias=True), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True)
        )
        # 确保 cond==0 时等价于"空条件"
        for m in self.cond_proj:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

        film_cond_dim = cond_dim * 2  # 拼 [t_emb, c_emb]

        # 自条件输入投影：两种路径
        self.in_proj = nn.Linear(x_dim, widths[0])
        self.sc_proj = nn.Linear(x_dim * 2, widths[0])

        # Down/Up 投影与块
        self.down_proj = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.down_proj.append(nn.Linear(widths[i], widths[i+1]))
            self.down_blocks.append(nn.ModuleList([
                ResMLPBlock(widths[i+1], film_cond_dim, hidden=widths[i+1], use_zero_mlp=use_zero_mlp)
                for _ in range(n_blocks_per_stage)
            ]))

        self.mid_blocks = nn.ModuleList([
            ResMLPBlock(widths[-1], film_cond_dim, hidden=widths[-1], use_zero_mlp=use_zero_mlp)
            for _ in range(n_blocks_per_stage)
        ])

        self.up_proj = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(widths) - 1)):
            self.up_proj.append(nn.Linear(widths[i+1]*2, widths[i]))  # 先 concat 再投影
            self.up_blocks.append(nn.ModuleList([
                ResMLPBlock(widths[i], film_cond_dim, hidden=widths[i], use_zero_mlp=use_zero_mlp)
                for _ in range(n_blocks_per_stage)
            ]))

        self.out = nn.Linear(widths[0], x_dim)

    def forward(self, x_t: Tensor, t: Tensor, c: Tensor = None, x0_sc: Tensor = None) -> Tensor:
        """
        x_t: [B, x_dim]; t: [B]; c: [B, cond_in_dim] or None; x0_sc: [B, x_dim] or None
        输出: v̂ 或 ε̂ （与 out_param 一致）
        """
        B = x_t.size(0)

        # 时间/条件嵌入
        t_emb = self.time_mlp(t)                         # [B, cond_dim]
        if c is None:
            c_emb = torch.zeros_like(t_emb)
        else:
            c_emb = self.cond_proj(c)                    # [B, cond_dim]
        cond_vec = torch.cat([t_emb, c_emb], dim=-1)     # [B, 2*cond_dim]

        # 自条件：把 x0_sc 与 x_t 拼接再投影
        if self.self_condition and (x0_sc is not None):
            h = self.sc_proj(torch.cat([x_t, x0_sc], dim=-1))
        else:
            h = self.in_proj(x_t)

        # Down
        skips = []
        for proj, blocks in zip(self.down_proj, self.down_blocks):
            h = proj(h)
            for blk in blocks:
                h = blk(h, cond_vec)
            skips.append(h)

        # Mid
        for blk in self.mid_blocks:
            h = blk(h, cond_vec)

        # Up（注意 reversed 对应）
        for proj, blocks, skip in zip(self.up_proj, self.up_blocks, reversed(skips)):
            h = torch.cat([h, skip], dim=-1)
            h = proj(h)
            for blk in blocks:
                h = blk(h, cond_vec)

        out = self.out(h)  # [B, x_dim]
        return out  # 作为 v̂ 或 ε̂

class LDM(nn.Module):
    def __init__(
        self,
        device,
        latent_dim: int,
        timesteps: int,
        cond_dim: int = None,
        predict: str = 'v',            # 'v' 或 'eps'，推荐 'v'
        unit_sphere: bool = False,     # 若嵌入需 L2 归一
        self_condition: bool = True,   # 是否启用自条件
        widths=(128, 256, 512),
        n_blocks_per_stage: int = 2,
        use_zero_mlp: bool = True,
    ):
        super().__init__()
        assert predict in ('v', 'eps')
        self.timesteps = timesteps
        self.device = device
        self.predict = predict
        self.unit_sphere = unit_sphere
        self.self_condition = self_condition

        betas = cosine_alpha_bar(timesteps, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

        self.net = VectorUNet(
            x_dim=latent_dim,
            widths=widths,
            time_dim=256,
            cond_dim=256,
            n_blocks_per_stage=n_blocks_per_stage,
            use_zero_mlp=use_zero_mlp,
            out_param=predict,
            self_condition=self_condition,
            cond_in_dim=cond_dim or latent_dim
        )

    # ---------- 基本运算 ----------
    def addnoise(self, x0: Tensor, t: Tensor):
        a_bar = self.alpha_bars[t].unsqueeze(-1)  # [B,1]
        eps = torch.randn_like(x0)
        x_t = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * eps
        if self.unit_sphere:
            x_t = F.normalize(x_t, dim=1)
        return x_t, eps

    def _v_to_eps(self, x_t: Tensor, t: Tensor, v_hat: Tensor):
        a_bar = self.alpha_bars[t].unsqueeze(-1)
        # 正确换元: ε̂ = sqrt(1-a_bar)*x_t + sqrt(a_bar)*v̂
        eps_hat = (1 - a_bar).sqrt() * x_t + a_bar.sqrt() * v_hat
        return eps_hat

    def _x0_from_v(self, x_t: Tensor, t: Tensor, v_hat: Tensor):
        a_bar = self.alpha_bars[t].unsqueeze(-1)
        # x0_hat = sqrt(a_bar)*x_t - sqrt(1-a_bar)*v̂
        x0_hat = a_bar.sqrt() * x_t - (1 - a_bar).sqrt() * v_hat
        if self.unit_sphere:
            x0_hat = F.normalize(x0_hat, dim=1)
        return x0_hat

    def _x0_from_eps(self, x_t: Tensor, t: Tensor, eps_hat: Tensor):
        a_bar = self.alpha_bars[t].unsqueeze(-1)
        # x0_hat = (x_t - sqrt(1 - a_bar) * eps_hat) / sqrt(a_bar)
        x0_hat = (x_t - (1 - a_bar).sqrt() * eps_hat) / a_bar.clamp_min(1e-12).sqrt()
        if self.unit_sphere:
            x0_hat = F.normalize(x0_hat, dim=1)
        return x0_hat

    # ---------- 训练: 自条件 + CFG 训练 + v-param ----------
    def loss(self, x0: Tensor, cond: Tensor = None, p_uncond: float = 0.1,
             lambda_proto: float = 0.0, proto: Tensor = None):
        """
        预训练: cond=None, lambda_proto=0
        微调:   cond=类原型, lambda_proto>0
        """
        B = x0.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x0.device, dtype=torch.long)
        x_t, eps = self.addnoise(x0, t)

        # ------ 第一趟估计 x0_hat 作为自条件（停止梯度） ------
        if self.self_condition:
            with torch.no_grad():
                out1 = self.net(x_t, t, c=None, x0_sc=None)
                x0_sc = self._x0_from_v(x_t, t, out1) if self.predict == 'v' else self._x0_from_eps(x_t, t, out1)
        else:
            x0_sc = None

        # ------ 条件丢弃（CFG 训练） ------
        if cond is None:
            cond_in = None
        else:
            drop = (torch.rand(B, device=x0.device) < p_uncond).float().unsqueeze(-1)  # [B,1]
            cond_in = cond * (1.0 - drop)

        # ------ 第二趟（自条件 50% 生效） ------
        if self.self_condition and (x0_sc is not None):
            use_sc = (torch.rand(B, device=x0.device) < 0.5)
            x0_sc_eff = torch.where(use_sc.unsqueeze(1), x0_sc, torch.zeros_like(x0_sc))
        else:
            x0_sc_eff = None

        out = self.net(x_t, t, c=cond_in, x0_sc=x0_sc_eff)  # v̂ 或 ε̂

        # ------ 主损失（v 或 ε） ------
        if self.predict == 'eps':
            target = eps
            L_noise = F.mse_loss(out, target)
            x0_hat = self._x0_from_eps(x_t, t, out)
        else:  # v-param
            a_bar = self.alpha_bars[t].unsqueeze(-1)
            v_target = a_bar.sqrt() * eps - (1 - a_bar).sqrt() * x0
            L_noise = F.mse_loss(out, v_target)
            x0_hat = self._x0_from_v(x_t, t, out)

        # ------ 原型一致性正则（微调才启用） ------
        if lambda_proto > 0.0 and (x0_hat is not None) and (proto is not None):
            # proto: [B, D] or [num_classes, D] 外面可先索引到 batch
            if self.unit_sphere:
                x0h = F.normalize(x0_hat, dim=1)
                pc  = F.normalize(proto, dim=1)
                L_proto = lambda_proto * (1.0 - (x0h * pc).sum(dim=-1)).mean()
            else:
                L_proto = lambda_proto * F.mse_loss(x0_hat, proto)
        else:
            L_proto = x0.new_zeros(())

        return L_noise + L_proto

    # ---------- 采样: CFG + v-param 公式修正 ----------
    @torch.no_grad()
    def _predict_model_out(self, x_t: Tensor, t: Tensor, cond: Tensor,
                           guidance: float = 0.0, x0_sc: Tensor = None):
        """
        推理期同样使用 self-conditioning：
        - 无引导/无条件：直接带上 x0_sc 调一次
        - CFG：有/无条件分支都共享同一个 x0_sc
        返回与 self.predict 一致的输出（'v' 或 'eps'）
        """
        if guidance <= 1e-6 or cond is None:
            return self.net(x_t, t, c=cond, x0_sc=x0_sc)
        out_c = self.net(x_t, t, c=cond, x0_sc=x0_sc)
        out_u = self.net(x_t, t, c=None, x0_sc=x0_sc)
        return out_u + guidance * (out_c - out_u)

    @torch.no_grad()
    def sample_backward_step(self, x_t: Tensor, t: int, cond: Tensor, guidance: float = 0.0,
                             simple_var: bool = True, temp: float = 1.0,
                             x0_sc: Tensor = None):
        B = x_t.size(0)
        t_tensor = torch.full((B,), t, dtype=torch.long, device=x_t.device)
        model_out = self._predict_model_out(x_t, t_tensor, cond,
                                            guidance=guidance, x0_sc=x0_sc)

        # 统一得到 ε̂
        if self.predict == 'eps':
            eps_hat = model_out
            x0_hat  = self._x0_from_eps(x_t, t_tensor, eps_hat)
        else:
            eps_hat = self._v_to_eps(x_t, t_tensor, model_out)
            x0_hat  = self._x0_from_v(x_t, t_tensor, model_out)

        # 计算均值与方差（DDPM）
        if t == 0:
            noise = 0.0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = (temp * torch.randn_like(x_t)) * var.sqrt()

        alpha_t = self.alphas[t]
        mean = (1 / alpha_t.sqrt()) * (x_t - (self.betas[t] / (1 - self.alpha_bars[t]).sqrt()) * eps_hat)
        x_prev = mean + noise
        if self.unit_sphere:
            x_prev = F.normalize(x_prev, dim=1)
        # 返回 x_prev 以及本步估计的 x0_hat，供下一步当做 self-conditioning
        return x_prev, x0_hat

    @torch.no_grad()
    def sample(self, shape, cond: Tensor = None, guidance: float = 0.0,
               simple_var: bool = True, temp: float = 1.0,
               init_match_radius: torch.Tensor = None, use_self_condition: bool = True):
        # 一次性日志确认参数传入
        if not hasattr(self, "_log_once"):
            print(f"[LDM.sample] temp={temp}, simple_var={simple_var}, guidance={guidance}, "
                  f"init_match_radius={'yes' if init_match_radius is not None else 'no'}")
            self._log_once = True
        
        z = torch.randn(shape, device=self.device)
        # （可选）用真实 z 半径对齐初始化
        if init_match_radius is not None:
            # 把 z 归一化到单位球，再乘以给定半径
            z = z / (z.norm(dim=1, keepdim=True) + 1e-12)
            z = z * init_match_radius.view(-1, 1).expand_as(z)
        
        if self.unit_sphere:
            z = F.normalize(z, dim=1)

        # ✅ 首步用全零自条件；之后每步用上一轮 x0_hat
        x0_sc = torch.zeros(shape, device=self.device) if use_self_condition else None

        for t in reversed(range(self.timesteps)):
            z, x0_hat = self.sample_backward_step(
                z, t, cond, guidance=guidance, simple_var=simple_var, temp=temp,
                x0_sc=x0_sc
            )
            if use_self_condition:
                x0_sc = x0_hat  # 传给下一步
        if self.unit_sphere:
            z = F.normalize(z, dim=1)
        return z

# =========================
# 辅助函数：微调参数过滤器
# =========================
def finetune_param_filter(name: str) -> bool:
    """
    微调时只训练小分支的参数过滤器
    返回 True 表示该参数需要训练
    """
    keys = ["film", "FiLM", "cond_proj", "zero", "lora_A", "lora_B", "gate"]
    return any(k in name for k in keys)