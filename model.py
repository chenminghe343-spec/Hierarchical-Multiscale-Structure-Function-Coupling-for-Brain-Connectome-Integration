import torch
import torch.nn as nn
from torch_geometric.nn.dense import DenseGCNConv, DenseGINConv, DenseSAGEConv
from Coupling import Parallel_Attention, Hie_Couple, GCL, CC_loss
from Pooling import MPCPool, dmon_loss_from_S
  
    
class HiM-SFC(nn.Module):
    def __init__(self, in_dim, hid=64, Pool_ratio=0.1, dropout=0.5, if_unsupervised=False, warm_ep=80):
        super().__init__()
        self.if_unsup = if_unsupervised

        self.sc_gcn1 = DenseGCNConv(in_dim, hid)
        self.sc_pool1 = MPCPool(in_dim=hid, use_entmax=True)
        self.sc_gcn2 = DenseGCNConv(hid, hid)
        self.fc_gcn1 = DenseGCNConv(in_dim, hid)
        self.fc_pool1 = MPCPool(in_dim=hid, use_entmax=True)
        self.fc_gcn2 = DenseGCNConv(hid, hid)


        self.paral_attn1 = Parallel_Attention(hid, dropout)
        self.hier_attn2 = Hie_Couple(hid, dropout)

        self.pr = Pool_ratio

        self.gcl_1 = GCL(hid)
        self.gcl_2 = GCL(hid)

        self.warm_ep = warm_ep


    def forward(self, sc, fc, xs, xf, epoch=0):
        xs1 = torch.relu(self.sc_gcn1(xs, sc))
        xf1 = torch.relu(self.fc_gcn1(xf, fc))

        # GCL & Coupling
        hs1, hf1, C1 = self.paral_attn1(xs1, xf1)

        xs2, sc2, S_s1 = self.sc_pool1(sc, xs1, ratio=self.pr)
        xf2, fc2, S_f1 = self.fc_pool1(fc, xf1, ratio=self.pr)

        # GNN embedding
        xs2 = torch.relu(self.sc_gcn2(xs2, sc2))
        xf2 = torch.relu(self.fc_gcn2(xf2, fc2))

        # GCL & Hierarchical coupling
        hs2, hf2, _ = self.hier_attn2(xs2, xf2, xs1, xf1, S_s1, S_f1)

        
        if self.if_unsup:
            g_loss = torch.tensor(0.0, device=sc.device, dtype=torch.float32)
            cc_loss = torch.tensor(0.0, device=sc.device, dtype=torch.float32)
            loss_s1, Qs1 = dmon_loss_from_S(sc, S_s1)
            loss_f1, Qf1 = dmon_loss_from_S(fc, S_f1)
            loss = 0.5 * (loss_s1 + loss_f1)
            if self.warm_ep < epoch:
                G_loss1 = self.gcl_1(xs1, xf1)
                G_loss2 = self.gcl_2(xs2, xf2)
                g_loss = 0.5 * (G_loss1 + G_loss2)
                # coupling-guided
                cc_loss = CC_loss(S_s1, S_f1, C1)
                return loss, g_loss, cc_loss, Qs1, Qf1, C1, (S_s1, S_f1)
            else:
                return loss, g_loss, cc_loss, Qs1, Qf1, C1, (S_s1, S_f1)
        else:
            G_loss1 = self.gcl_1(xs1, xf1)
            G_loss2 = self.gcl_2(xs2, xf2)
            g_loss = 0.5 * (G_loss1 + G_loss2)
            cc_loss = CC_loss(S_s1, S_f1, C1)
            loss_s1, Qs1 = dmon_loss_from_S(sc, S_s1)
            loss_f1, Qf1 = dmon_loss_from_S(fc, S_f1)
            loss = 0.5 * (loss_s1 + loss_f1)
            # multimodal global pooling
            h1 = torch.cat([hs1, hf1], dim=-1)         # [B, 2*hid]
            h2 = torch.cat([hs2, hf2], dim=-1)         # [B, 2*hid]
            return (h1, h2), loss, g_loss, cc_loss, Qs1, Qf1, C1, (S_s1, S_f1)


class MulProjHead2(nn.Module):
    def __init__(self, hid, dropout, task='reg'):
        super().__init__()

        self.proj1 = nn.Sequential(
            nn.Linear(hid*2, hid), 
            nn.ReLU(), 
            nn.LayerNorm(hid)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(hid*2, hid), 
            nn.ReLU(), 
            nn.LayerNorm(hid)
        )

        self.readout_mlp = nn.Sequential(
            nn.Linear(hid * 2, hid), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hid, 1)
        )

        self.task = task


    def forward(self, h):
        h1, h2 = h

        h1 = self.proj1(h1)
        h2 = self.proj2(h2)
        # h3 = self.proj3(h3)

        # global mean pool
        h1 = h1.mean(dim=1)
        h2 = h2.mean(dim=1)
        # h3 = h3.mean(dim=1)
        out = self.readout_mlp(torch.cat([h1, h2], dim=-1))
        if self.task == 'cls':
            y = torch.sigmoid(out)
        elif self.task == 'reg':
            y = out
        return y.reshape(y.shape[0])      
    

class MulProjHead3(nn.Module):
    def __init__(self, hid, dropout, task='reg'):
        super().__init__()

        self.proj1 = nn.Sequential(
            nn.Linear(hid*2, hid), 
            nn.ReLU(), 
            nn.LayerNorm(hid)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(hid*2, hid), 
            nn.ReLU(), 
            nn.LayerNorm(hid)
        )
        self.proj3 = nn.Sequential(
            nn.Linear(hid*2, hid), 
            nn.ReLU(), 
            nn.LayerNorm(hid)
        )

        self.readout_mlp = nn.Sequential(
            nn.Linear(hid * 3, hid), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hid, 1)
        )

        self.task = task


    def forward(self, h):
        h1, h2, h3 = h

        h1 = self.proj1(h1)
        h2 = self.proj2(h2)
        h3 = self.proj3(h3)

        # global mean pool
        h1 = h1.mean(dim=1)
        h2 = h2.mean(dim=1)
        h3 = h3.mean(dim=1)
        out = self.readout_mlp(torch.cat([h1, h2, h3], dim=-1))
        if self.task == 'cls':
            y = nn.Sigmoid(out)
        elif self.task == 'reg':
            y = out
        return y.reshape(y.shape[0])