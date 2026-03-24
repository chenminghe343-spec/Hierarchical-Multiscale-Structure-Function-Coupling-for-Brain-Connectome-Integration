import torch.nn as nn
import torch
import torch.nn.functional as F
from entmax import entmax15
import math

def dmon_loss_from_S(
    adj: torch.Tensor,      # [B, N, N]
    S: torch.Tensor,        # [B, N, K]
    collapse_lambda: float = 0.1,
    eps: float = 1e-12,
    reduce: str = "mean",
):
    assert adj.dim() == 3 and S.dim() == 3
    B, N, _ = adj.shape
    _, Ns, K = S.shape
    assert Ns == N,

    deg = adj.sum(dim=-1)
    two_m = deg.sum(dim=-1).clamp_min(eps)

    AS = torch.matmul(adj, S)
    StAS_trace = (S * AS).sum(dim=(1, 2))

    Stk = torch.matmul(S.transpose(1, 2), deg.unsqueeze(-1)).squeeze(-1)
    corr = (Stk.pow(2).sum(dim=1) / two_m).clamp_min(0.0)
    spectral = - (StAS_trace - corr) / two_m

    sizes = S.sum(dim=1)
    collapse = (sizes.norm(dim=1) / max(N, 1) * math.sqrt(K) - 1.0)
    collapse = collapse.clamp_min(0.0)

    per_graph = spectral + collapse_lambda * collapse

    if reduce == "mean":
        loss = per_graph.mean()
    elif reduce == "sum":
        loss = per_graph.sum()
    elif reduce == "none":
        loss = per_graph
    else:
        raise ValueError("reduce must be 'mean' | 'sum' | 'none'")

    Q = -spectral.mean().detach()
    return loss, Q

def batched_topk_select(x, scores, k):
    B, N, D = x.shape
    k = max(1, min(k, N))
    vals, idx = torch.topk(scores, k=k, dim=1, largest=True)
    idx_exp = idx.unsqueeze(-1).expand(B, k, D)
    P = torch.gather(x, dim=1, index=idx_exp)
    return P, idx

class MPCPool(nn.Module):
    def __init__(
        self,
        in_dim,
        conv_hidden=None,
        mlp_hidden=None,
        temperature=1.0,
        use_entmax=True,
        use_centrality=True,
        degree_emb_dim=None,
        use_weighted_degree=False,
        score_activation=None,
    ):
        super().__init__()
        C = in_dim
        conv_out = conv_hidden or C
        mlp_h = mlp_hidden or C

        self.use_centrality = use_centrality
        self.use_weighted_degree = use_weighted_degree
        self.score_activation = score_activation

        if self.use_centrality:
            self.degree_emb_dim = degree_emb_dim or (C // 2 if C >= 2 else 1)
            self.degree_mlp = nn.Sequential(
                nn.Linear(1, self.degree_emb_dim),
                nn.GELU(),
                nn.Linear(self.degree_emb_dim, self.degree_emb_dim),
            )
            conv_in_channels = C + self.degree_emb_dim
        else:
            self.degree_emb_dim = 0
            self.degree_mlp = None
            conv_in_channels = C

        self.conv = nn.Conv1d(
            in_channels=conv_in_channels,
            out_channels=conv_out,
            kernel_size=1,
            bias=True,
        )
        self.norm = nn.LayerNorm(conv_out)

        self.mlp = nn.Sequential(
            nn.Linear(conv_out, mlp_h),
            nn.GELU(),
            nn.Linear(mlp_h, 1)
        )

        self.tau = temperature
        self.use_entmax = use_entmax

    def _compute_degree_emb(self, adj):
        if not self.use_centrality or self.degree_mlp is None:
            return None

        if self.use_weighted_degree:
            deg = adj.sum(-1)
        else:
            deg = (adj > 0).float().sum(-1)  # [B, N]

        deg_max = deg.max(dim=-1, keepdim=True).values + 1e-6
        deg_norm = deg / deg_max

        deg_feat = deg_norm.unsqueeze(-1)
        degree_emb = self.degree_mlp(deg_feat)
        return degree_emb

    def forward(self, adj, x, ratio=None, k=None):
        B, N, D = x.shape
        if k is None:
            assert ratio is not None and 0 < ratio <= 1.0, "pls set ratio"
            k = max(1, int(round(N * ratio)))
        else:
            k = max(1, min(k, N))

        if self.use_centrality and self.degree_mlp is not None:
            degree_emb = self._compute_degree_emb(adj)
            x_multi = torch.cat([x, degree_emb], dim=-1)
        else:
            x_multi = x
        h = self.conv(x_multi.transpose(1, 2))
        h = h.transpose(1, 2)
        h = self.norm(h)

        score = self.mlp(h).squeeze(-1)          # [B, N]

        if self.score_activation is not None:
            act = self.score_activation.lower()
            if act == "relu":
                score = F.relu(score)
            elif act == "softplus":
                score = F.softplus(score)
            elif act == "tanh":
                score = torch.tanh(score)
            elif act == "sigmoid":
                score = torch.sigmoid(score)
            else:
                raise ValueError(f"Unsupported score_activation: {self.score_activation}")

        P, idx = batched_topk_select(x, score, k)

        logits = torch.matmul(x, P.transpose(1, 2)) / math.sqrt(D)
        if self.tau and self.tau > 0:
            logits = logits / self.tau

        if self.use_entmax:
            S = entmax15(logits, dim=-1)
        else:
            S = F.softmax(logits, dim=-1)

        x_ = torch.matmul(S.transpose(1, 2), x)
        adj_ = torch.matmul(torch.matmul(S.transpose(1, 2), adj), S) 

        return x_, adj_, S
    


