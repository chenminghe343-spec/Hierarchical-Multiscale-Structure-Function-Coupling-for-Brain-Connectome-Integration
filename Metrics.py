from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)
from scipy.stats import pearsonr
import torch.nn as nn
import numpy as np



def evaluation_metrics(task: str):
    if task == "cls":
        loss_fn = nn.BCELoss()

        def metric_fn(y_true, y_pred):
            acc = accuracy_score(y_true, y_pred > 0.5)
            auc = roc_auc_score(y_true, y_pred)

            sen = recall_score(y_true, y_pred > 0.5)
            spec = recall_score(y_true, y_pred > 0.5, pos_label=0)
            return {
                "acc": acc, "auc": auc, "sen": sen, "spec": spec,
            }
    elif task == "reg":
        loss_fn = nn.MSELoss()

        def metric_fn(y_true, y_pred):
            y_true = np.array(y_true).reshape(-1)
            y_pred = np.array(y_pred).reshape(-1)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            pearson_corr, _ = pearsonr(y_true, y_pred)
            return {"mae": mae, "rmse": rmse, "pcc": pearson_corr}
    else:
        raise ValueError("Invalid task type. Choose 'cls' or 'reg'.")

    return loss_fn, metric_fn
