import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from typing import Callable
import numpy as np
import random
import math
from PrepareData import MulSparseDataset
from model import HiM-SFC,  MulProjHead2
from Metrics import *



def train_unsup(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    g_loss = 0.0
    cc_loss = 0.0
    Qs1_loss = 0.0
    Qf1_loss = 0.0
    total = 0
    beta = 0.1
    lamd = 0.11
    for sc, fc, xs, xf, y in loader:
        sc = sc.to(device)
        fc = fc.to(device)
        xs = xs.to(device)
        xf = xf.to(device)
        loss_dm, G_loss, CC_loss, Qs1, Qf1, _, _ = model(sc, fc, xs, xf, epoch)
        loss = loss_dm + beta * G_loss + lamd * CC_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        g_loss += G_loss.item() * bs
        cc_loss += CC_loss.item() * bs
        Qs1_loss += Qs1.item() * bs
        Qf1_loss += Qf1.item() * bs
        total += bs
    
    return total_loss / max(total, 1), g_loss / max(total, 1), cc_loss / max(total, 1), Qs1_loss / max(total, 1), Qf1_loss / max(total, 1)


def train_one_epoch(ep, model, head, loader, optimizer, device, task, criterion):
    al, be, ga = 0.01, 0.1, 0.11
    model.train()
    head.train()
    total_loss = 0.0
    Qs1_loss = 0.0
    Qf1_loss = 0.0
    total = 0
    for sc, fc, xs, xf, y in loader:
        sc = sc.to(device)
        fc = fc.to(device)
        xs = xs.to(device)
        xf = xf.to(device)
        y = y.to(device)

        h, loss_dm, G_loss, CC_loss, Qs1, Qf1, _, _  = model(sc, fc, xs, xf)
        out = head(h)

        loss = criterion(out, y)
        loss = loss + al * loss_dm + be * G_loss + ga * CC_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        Qs1_loss += Qs1.item() * bs
        Qf1_loss += Qf1.item() * bs
        total += bs
    
    return total_loss / max(total, 1), Qs1_loss / max(total, 1), Qf1_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, head, loader, device, task):
    model.eval()
    head.eval()
    total = 0
    if task == "cls":
        correct = 0
        for sc, fc, xs, xf, y in loader:
            sc = sc.to(device)
            fc = fc.to(device)
            xs = xs.to(device)
            xf = xf.to(device)
            y = y.to(device)
            h = model(sc, fc, xs, xf, if_unsupervised=False)
            logits = head(h)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        acc = correct / max(total, 1)
        return {"acc": acc}
    else:
        mae_sum = 0.0
        mse_sum = 0.0

        all_preds = []
        all_y = []

        for sc, fc, xs, xf, y in loader:
            sc = sc.to(device)
            fc = fc.to(device)
            xs = xs.to(device)
            xf = xf.to(device)
            y = y.to(device)
            h, _, _, _, _, _, _, _ = model(sc, fc, xs, xf)
            preds = head(h)
            if preds.dim() == 1:
                preds = preds.unsqueeze(-1)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            mae_sum += torch.nn.functional.l1_loss(preds, y, reduction="sum").item()
            mse_sum += torch.nn.functional.mse_loss(preds, y, reduction="sum").item()
            total += y.numel()

            all_preds.append(preds.detach().reshape(-1).cpu())
            all_y.append(y.detach().reshape(-1).cpu())

        mae = mae_sum / max(total, 1)
        rmse = math.sqrt(mse_sum / max(total, 1))
        p = torch.cat(all_preds)
        t = torch.cat(all_y)
        p = p - p.mean()
        t = t - t.mean()
        denom = (p.std(unbiased=False) * t.std(unbiased=False)).clamp_min(1e-12)
        pcc = (p * t).mean() / denom
        pcc = pcc.item()
        return {"mae": mae, "rmse": rmse, "pcc": pcc}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def kfold_cross_validation(model_name: Callable, head_name: Callable, model_path: str, head_path: str,sc_path: str, fc_path: str, 
                           label_path: str, in_dim=400, hid_dim=64, pool_ratio=0.1,  k_folds: int = 5, batch_size: int = 32, 
                           un_epoch: int=50, epoch: int = 10, learning_rate: float = 1e-3, dropout_rate = 0.5, task: str = "reg", seed = 20, val_ratio=0.2, warm_ep=80):

    setup_seed(seed)
    if task == 'cls':
        _k_fold_val_metrics = {'best_on_acc': {'acc':[], 'auc':[], 'f1':[], 'sen':[], 'spec':[], 'pre':[]},
                    'best_on_auc': {'acc':[], 'auc':[], 'f1':[], 'sen':[], 'spec':[]}}
    elif task == 'reg':
        _k_fold_val_metrics = {'best_on_mae': {'mae':[], 'rmse':[], 'pcc':[]},
                           'best_on_rmse': {'mae':[], 'rmse':[], 'pcc':[]},
                           'best_on_pcc': {'mae':[], 'rmse':[], 'pcc':[]}}
    else:
        raise ValueError("Invalid task type. Choose 'gender' or 'age'.")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Enter Main")
    print("Data Loading...")
    dataset = MulSparseDataset(sc_path, fc_path, label_path, task)
    print("Loading Finished...")

    print('====================================================Stage one: Self-supervised learning==========================================\n')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model_name(in_dim, hid_dim, pool_ratio, dropout_rate, if_unsupervised=True, warm_ep=warm_ep).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    best_Qs1 = float('inf')
    best_Qf1 = float('inf')
    best_g = float('inf')
    best_cc = float('inf')

    for ep in range(1, un_epoch+1):
        total_loss, g_loss, cc_loss, Qs1, Qf1 = train_unsup(model, dataloader, optimizer, device, ep)
        if total_loss < best_loss and ep > warm_ep :
            best_loss = total_loss
            best_Qs1 = Qs1
            best_Qf1 = Qf1
            best_g = g_loss
            best_cc = cc_loss
            best_ep = ep
            torch.save(model.state_dict(), model_path+'best_model_HCP.pth')

        if ep % 5 == 0 or ep == 1 or ep == un_epoch:
            print(f"Unsupervised Learning -> Epoch {ep:03d} | Total_loss={total_loss:.4f} | G_loss={g_loss:.4f} | CC_loss={cc_loss:.4f} | Qs1_loss={Qs1:.4f} | Qf1_loss={Qf1:.4f}")# | Q2_loss={Q2:.4f}")

    print(f"Best Performance -> Epoch {best_ep:03d} | Total_loss={best_loss:.4f} | G_loss={best_g:.4f} | CC_loss={best_cc:.4f} | Qs1_loss={best_Qs1:.4f} | Qf1_loss={best_Qf1:.4f}")


    print('====================================================Stage two: Supervised learning==========================================\n')
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    loss_fn, metric_fn = evaluation_metrics(task)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset.Get_Index())):
        print(f"Fold {fold + 1}/{k_folds}")

        if task == 'cls':
            best_metrics = {'best_on_acc': {'acc':0, 'auc':0, 'f1':0, 'sen':0, 'spec':0, 'pre':0},
                            'best_on_auc': {'acc':0, 'auc':0, 'f1':0, 'sen':0, 'spec':0}}
        elif task == 'reg':
            best_metrics = {'best_on_mae': {'mae':100, 'rmse':100, 'pcc':-1},
                            'best_on_rmse': {'mae':100, 'rmse':100, 'pcc':-1},
                            'best_on_pcc': {'mae':100, 'rmse':100, 'pcc':-1}}
            

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = model_name(in_dim, hid_dim, pool_ratio, dropout_rate, if_unsupervised=False).to(device)
        model.load_state_dict(torch.load(model_path+'best_model_HCP.pth'))
        classifier = head_name(hid_dim, dropout_rate).to(device)
        optimizer = optim.Adam([{"params": classifier.parameters(), "lr":learning_rate},
                                 {"params": model.parameters(), "lr":learning_rate * 0.05},])

        for ep in range(epoch):
            train_loss, Qs1, Qf1 = train_one_epoch(ep, model, classifier, train_loader, optimizer, device, task,  loss_fn)
            val_metric = evaluate(model, classifier, val_loader, device, task)

            if task == 'reg':
                if val_metric['mae'] < best_metrics['best_on_mae']['mae']:
                    best_metrics['best_on_mae'] = val_metric.copy()
                    torch.save(classifier.state_dict(),
                            head_path + f'best_model_on_mae_fold{fold+1}.pth')
                    torch.save(model.state_dict(),
                            head_path + 'best_encoder_HCP_on_mae_fold' + str(fold+1) + '.pth')

                if val_metric['pcc'] > best_metrics['best_on_pcc']['pcc']:
                    best_metrics['best_on_pcc'] = val_metric.copy()
                    torch.save(classifier.state_dict(),
                            head_path + f'best_model_on_pcc_fold{fold+1}.pth')
                    torch.save(model.state_dict(),
                            head_path + 'best_encoder_HCP_on_pcc_fold' + str(fold+1) + '.pth')
                if val_metric['rmse'] < best_metrics['best_on_rmse']['rmse']:
                    best_metrics['best_on_rmse'] = val_metric.copy()
                    torch.save(classifier.state_dict(),
                            head_path + f'best_model_on_rmse_fold{fold+1}.pth')
                    torch.save(model.state_dict(),
                            head_path + 'best_encoder_HCP_on_rmse_fold' + str(fold+1) + '.pth')

            if task == 'cls':
                if val_metric['acc'] > best_metrics['best_on_acc']['acc']:
                    best_metrics['best_on_acc'] = val_metric.copy()
                    torch.save(classifier.state_dict(),
                            model_path + '/' + task + '/' + 'best_classifier_on_acc_fold' + str(fold+1) + '.pth')
                    torch.save(model.state_dict(),
                            model_path + '/' + task + '/' + 'best_encoder_on_acc_fold' + str(fold+1) + '.pth')
                if val_metric['auc'] > best_metrics['best_on_auc']['auc']:
                    best_metrics['best_on_auc'] = val_metric.copy()
                    torch.save(classifier.state_dict(),
                            model_path + '/' + task + '/' + 'best_classifier_on_auc_fold' + str(fold+1) + '.pth')
                    torch.save(model.state_dict(),
                            model_path + '/' + task + '/' + 'best_encoder_on_auc_fold' + str(fold+1) + '.pth')

            if ep % 5 == 0 or ep == 1 or ep == epoch-1:
                print(f"Epoch {ep + 1}/{epoch}, Train Loss: {train_loss:.4f}, Qs1: {Qs1:.4f}, Qf1: {Qf1:.4f},  Val Metric: {val_metric}")

        print('\n')
        print('best val metric:')
        print(f'Fold {fold + 1}/{k_folds} best metrics: {best_metrics}')
        for standard in best_metrics:
            for metric in best_metrics[standard]:
                _k_fold_val_metrics[standard][metric].append(best_metrics[standard][metric])


    print('')
    print('Final Results:')
    print('Val results for 5 folds')
    for standard in _k_fold_val_metrics:
      print(f'{standard}: ')
      for metric in _k_fold_val_metrics[standard]:
          mean = np.mean(_k_fold_val_metrics[standard][metric])
          std = np.std(_k_fold_val_metrics[standard][metric])
          print(f"{metric} mean: {mean:.4f} +- std: {std:.4f}")


if __name__ == "__main__":

    kfold_cross_validation(
        model_name=HiM-SFC,
        head_name=MulProjHead2,
        model_path="/home/Unsup/",
        head_path="/home/Results/",
        sc_path="/media/Conn/sc_data.npy",
        fc_path="/media/Conn/fc_data.npy",
        label_path="/media/Conn/label_data.npy",
        in_dim=400,
        hid_dim=64,
        pool_ratio=0.12,
        k_folds=5,
        batch_size=32,
        un_epoch=300,
        epoch=300,
        learning_rate=1e-3,
        dropout_rate=0.5,
        task="reg",
        seed=42,
        val_ratio=0.2,
        warm_ep=80
    )