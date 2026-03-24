import torch
import torch.nn as nn
from entmax import entmax15
import math
import torch.nn.functional as F

def CC_loss(S_s, S_f, coupling, eps = 1e-3):
    coupling = coupling.abs()
    row_strength = coupling.sum(dim=-1)
    col_strength = coupling.sum(dim=-2)
    w = 0.5 * (row_strength + col_strength)
    w = w / (w.max(dim=-1, keepdim=True).values + 1e-8)

    kl1 = S_s * (torch.log(S_s + eps) - torch.log(S_f + eps))
    kl2 = S_f * (torch.log(S_f + eps) - torch.log(S_s + eps))
    kl_loss = 0.5 * (kl1 + kl2).sum(dim=-1) # [B, N]
    cg_loss = (w * kl_loss).mean()
    return cg_loss

class GCL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = None,
        readout: str = "mean",
        temperature: float = 0.1,
        use_all_negatives: bool = False,
    ):
        super().__init__()
        assert readout in ["mean", "max", "sum"], "readout must be mean/max/sum"

        self.readout = readout
        self.temperature = temperature
        self.use_all_negatives = use_all_negatives
        if hidden_dim == None:
            hidden_dim = in_dim
        self.proj_sc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.proj_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
        )


    def graph_readout(self, x: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            return x.mean(dim=1)
        elif self.readout == "max":
            return x.max(dim=1).values
        elif self.readout == "sum":
            return x.sum(dim=1)
        else:
            raise ValueError(f"Unknown readout mode: {self.readout}")

    def forward(self, sc_nodes: torch.Tensor, fc_nodes: torch.Tensor) -> torch.Tensor:
        assert sc_nodes.shape == fc_nodes.shape, "SC / FC tensor shape must match"
        device = sc_nodes.device

        g_sc = self.graph_readout(sc_nodes)  # [B, D]
        g_fc = self.graph_readout(fc_nodes)  # [B, D]

        z_sc = self.proj_sc(g_sc)  # [B, P]
        z_fc = self.proj_fc(g_fc)  # [B, P]

        z_sc = F.normalize(z_sc, p=2, dim=-1)
        z_fc = F.normalize(z_fc, p=2, dim=-1)

        B, P = z_sc.shape

        if not self.use_all_negatives:
            logits_sf = (z_sc @ z_fc.t()) / self.temperature  # [B, B]
            logits_fs = (z_fc @ z_sc.t()) / self.temperature  # [B, B]

            labels = torch.arange(B, device=device)

            loss_sf = F.cross_entropy(logits_sf, labels)
            loss_fs = F.cross_entropy(logits_fs, labels)
            loss = 0.5 * (loss_sf + loss_fs)
            return loss

        all_emb = torch.cat([z_fc, z_sc], dim=0) 

        logits_s = (z_sc @ all_emb.t()) / self.temperature  # [B, 2B]
        target_s = torch.arange(B, device=device)

        mask_s = torch.zeros_like(logits_s, dtype=torch.bool)
        idx = torch.arange(B, device=device)
        mask_s[idx, B + idx] = True
        logits_s = logits_s.masked_fill(mask_s, -1e9)

        loss_s = F.cross_entropy(logits_s, target_s)

        logits_f = (z_fc @ all_emb.t()) / self.temperature  # [B, 2B]
        target_f = B + torch.arange(B, device=device)

        mask_f = torch.zeros_like(logits_f, dtype=torch.bool)
        mask_f[idx, idx] = True
        logits_f = logits_f.masked_fill(mask_f, -1e9)

        loss_f = F.cross_entropy(logits_f, target_f)

        loss = 0.5 * (loss_s + loss_f)
        return loss


class Cross_Attention(nn.Module):
    def __init__(self, embed_dim, dropout=0.5):
        super(Cross_Attention, self).__init__()
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.LN = nn.LayerNorm(embed_dim)

    def forward(self, x_c, x_n):
        Q = self.Q(x_c) # [B, N2, D]
        K = self.K(x_n) # [B, N1, D]

        d_k = Q.size(-1)
        A = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # attention matrix [B, N2, N1]
        A = entmax15(A, dim=-1)

        A = self.dropout(A)

        return A

class Parallel_Attention(nn.Module):
    def __init__(self, embed_dim, dropout=0.5):
        super(Parallel_Attention, self).__init__()
        self.Qs = nn.Linear(embed_dim, embed_dim)
        self.Kf = nn.Linear(embed_dim, embed_dim)
        self.Vs = nn.Linear(embed_dim, embed_dim)
        self.Vf = nn.Linear(embed_dim, embed_dim)
        self.Os = nn.Linear(embed_dim, embed_dim)
        self.Of = nn.Linear(embed_dim, embed_dim)
        self.s_norm = nn.LayerNorm(embed_dim)
        self.f_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs, xf):
        Q = self.Qs(xs)
        K = self.Kf(xf)
        Vs = self.Vs(xs)
        Vf = self.Vf(xf)

        d_k = Q.size(-1)
        A = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # attention matrix
        A_s = F.entmax15(A, dim=-1)
        A_f = F.entmax15(A, dim=-2)

        if self.dropout is not None:
            A_s = self.dropout(A_s)
            A_f = self.dropout(A_f)


        xs_ = self.dropout(self.Os(torch.matmul(A_s, Vf)))
        xf_ = self.dropout(self.Of(torch.matmul(A_f.transpose(-2, -1), Vs)))

        xs_ = xs_ + self.s_norm(xs)
        xs_ = xf_ + self.f_norm(xf)

        return xs_, xf_, A
    
class Hie_Couple(nn.Module):
    def __init__(self, embed_dim, dropout=0.5, bias_fusion='log'):
        super(Hie_Couple, self).__init__()
        self.Sup_F_Fine_S_Attn = Cross_Attention(embed_dim, dropout)
        self.Sup_S_Fine_F_Attn = Cross_Attention(embed_dim, dropout)

        self.Qs = nn.Linear(embed_dim, embed_dim)
        self.Kf = nn.Linear(embed_dim, embed_dim)
        self.Vs = nn.Linear(embed_dim, embed_dim)
        self.Vf = nn.Linear(embed_dim, embed_dim)
        self.Os = nn.Linear(embed_dim, embed_dim)
        self.Of = nn.Linear(embed_dim, embed_dim)
        self.s_norm = nn.LayerNorm(embed_dim)
        self.f_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.bias_fusion=bias_fusion

    def fuse_bias(self, C_M, bias):
        if self.bias_fusion == "log":
            bias_log = torch.log(bias.clamp(min=1e-6))
            return C_M + bias_log
        elif self.bias_fusion == "layernorm":
            bias_norm = F.layer_norm(bias, bias.shape[-1:])
            return C_M + bias_norm
        elif self.bias_fusion == "linear":
            if self.bias_linear is None:
                self.bias_linear = nn.Linear(bias.size(-1), bias.size(-1), bias=False).to(bias.device)
            bias_lin = self.bias_linear(bias)
            return C_M + bias_lin
    
    def forward(self, xs, xf, xs_fine, xf_fine, S_s, S_f):
        Q = self.Qs(xs)
        K = self.Kf(xf)
        Vs = self.Vs(xs)
        Vf = self.Vf(xf)

        d_k = Q.size(-1)
        C_M = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        C_M_F = self.Sup_F_Fine_S_Attn(xf, xs_fine) # [B, N2, N1]
        C_M_S = self.Sup_S_Fine_F_Attn(xs, xf_fine)

        bias_F = torch.matmul(C_M_F, S_s) # [B, N2, N2]
        bias_F = bias_F.transpose(-2, -1)
        bias_S = torch.matmul(C_M_S, S_f)

        A_s = self.fuse_bias(C_M, bias_S)
        A_s = entmax15(A_s, dim=-1)

        A_f = self.fuse_bias(C_M, bias_F)
        A_f = entmax15(A_f, dim=-2)

        if self.dropout is not None:
            A_s = self.dropout(A_s)
            A_f = self.dropout(A_f)

        xs_ = self.dropout(self.Os(torch.matmul(A_s, Vf)))
        xf_ = self.dropout(self.Of(torch.matmul(A_f.transpose(-2, -1), Vs)))

        xs_ = xs_ + self.s_norm(xs)
        xs_ = xf_ + self.f_norm(xf)

        return xs_, xf_, C_M




