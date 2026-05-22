import torch
import numpy as np
import pandas as pd
import math
import time
import json
import re
import glob
import sys
import os 
import random
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import ReLU
from copy import deepcopy

### UTILS 
def activation_helper(activation, dim=None):
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: %s' % activation)
    return act



def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def regularize(network, lam, penalty):
    '''
    Calculate regularization term for first layer weight matrix.

    Args:
      network: MLP network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)
    
    

def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])
   
    
class cMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation='relu'):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation)
            for _ in range(num_series)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if ignore_lag:
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC


def train_model_adam(cmlp, X, lr, max_iter, lam=0, lam_ridge=0, penalty='H',
                     lookback=5, check_every=100, verbose=1):
    '''Train model with Adam.'''
    lag = cmlp.lag
    p = X.shape[-1]
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(cmlp.parameters(), lr=lr)
    train_loss_list = []

    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None

    for it in range(max_iter):
        # Calculate loss.
        loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
                    for i in range(p)])

        # Add penalty terms.
        if lam > 0:
            loss = loss + sum([regularize(net, lam, penalty)
                               for net in cmlp.networks])
        if lam_ridge > 0:
            loss = loss + sum([ridge_regularize(net, lam_ridge)
                               for net in cmlp.networks])

        # Take gradient step.
        loss.backward()
        optimizer.step()
        cmlp.zero_grad()

        # Check progress.
        if (it + 1) % check_every == 0:
            mean_loss = loss / p
            train_loss_list.append(mean_loss.detach())

            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Loss = %f' % mean_loss)

            # Check for early stopping.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(cmlp)
            elif (it - best_it) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break

    # Restore best model.
    restore_parameters(cmlp, best_model)

    return train_loss_list


def eval_mse_after_train(cmlp, X): 
    lag = cmlp.lag
    p   = X.shape[-1]
    loss_fn = torch.nn.MSELoss(reduction='mean')
    with torch.no_grad():
        mse = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1]) for i in range(p)])
    return float((mse / p).cpu().numpy())

def jaccard(A, B):
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return inter / max(union, 1)

def avg_pairwise_jaccard(GCs):
    # GCs: list of (p,p) 0/1
    if len(GCs) < 2: return 1.0
    vals = []
    for i in range(len(GCs)):
        for j in range(i+1, len(GCs)):
            vals.append(jaccard(GCs[i], GCs[j]))
    return float(np.mean(vals)) if vals else 1.0


def _sample_unique_combos(grid, n_trials, seed=42):
    rng = np.random.RandomState(seed)
    keys = ["lag", "hidden", "lam", "lam_r", "lr"]

    all_sizes = [len(grid[k]) for k in keys]
    max_unique = int(np.prod(all_sizes))
    n = min(n_trials, max_unique)

    seen = set()
    combos = []
    while len(combos) < n:
        tup = tuple(rng.choice(grid[k]) for k in keys)
        if tup not in seen:
            seen.add(tup)
            combos.append(tup)
    return combos

def tune_ngc_random(
    normal_runs_X,               
    grid=None,
    n_trials=30,
    stability_weight=0.1,
    max_iter=3000,
    check_every=200,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    standardize=True,
    subsample_T=None,
    seed=42,
    min_edges=1,
    max_edges=None):
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    if grid is None:
        grid = {
            "lag":    [4, 8, 12, 16],
            "hidden": [32, 64, 128],
            "lam":    [1e-3, 2e-3],
            "lam_r":  [1e-4],
            "lr":     [1e-3, 5e-4],
        }

    combos = _sample_unique_combos(grid, n_trials=n_trials, seed=seed)
    results = []

    for (lag, hidden, lam, lam_r, lr) in combos:
        run_MSEs, run_GCs = [], []

        for X in normal_runs_X:
            X_arr = np.asarray(X, dtype=float)

            if subsample_T is not None and X_arr.shape[0] > subsample_T:
                idx = np.linspace(0, X_arr.shape[0] - 1, subsample_T).astype(int)
                X_arr = X_arr[idx]

            if standardize:
                mu = X_arr.mean(axis=0, keepdims=True)
                sd = X_arr.std(axis=0, keepdims=True) + 1e-12
                X_arr = (X_arr - mu) / sd

            T, p = X_arr.shape
            X_t = torch.tensor(X_arr, dtype=torch.float32, device=device).unsqueeze(0)

            model = cMLP(
                num_series=p,
                lag=int(lag),
                hidden=[int(hidden)],
                activation='relu'
            ).to(device)

            _ = train_model_adam(
                model, X_t,
                lr=float(lr),
                max_iter=int(max_iter),
                lam=float(lam),
                lam_ridge=float(lam_r),
                penalty='GL',
                check_every=int(check_every),
                lookback=5,
                verbose=0)

            mse = eval_mse_after_train(model, X_t)
            run_MSEs.append(mse)

            with torch.no_grad():
                S = model.GC(threshold=False).detach().cpu().numpy()

        mean_mse = float(np.mean(run_MSEs))
        stab = avg_pairwise_jaccard(run_GCs)
        score = mean_mse + stability_weight * (1.0 - stab)

        results.append({
            "hp": {
                "lag": lag,
                "hidden": hidden,
                "lam": lam,
                "lam_r": lam_r,
                "lr": lr
            },
            "mean_mse": mean_mse,
            "stability": stab,
            "score": score
        })

        print(
            f"[lag={lag}, hid={hidden}, lam={lam}, lam_r={lam_r}, lr={lr}]  "
            f"MSE={mean_mse:.6f}  stab={stab:.3f}  score={score:.6f}"
        )

    results = sorted(results, key=lambda d: d["score"])
    best = results[0]
    print("\nBest (random search):", best)

    lag = best["hp"]["lag"]
    hidden = best["hp"]["hidden"]
    lam = best["hp"]["lam"]
    lam_r = best["hp"]["lam_r"]
    lr = best["hp"]["lr"]

    S_list = []
    for X in normal_runs_X:
        X_arr = np.asarray(X, dtype=float)

        if subsample_T is not None and X_arr.shape[0] > subsample_T:
            idx = np.linspace(0, X_arr.shape[0] - 1, subsample_T).astype(int)
            X_arr = X_arr[idx]

        if standardize:
            mu = X_arr.mean(axis=0, keepdims=True)
            sd = X_arr.std(axis=0, keepdims=True) + 1e-12
            X_arr = (X_arr - mu) / sd

        T, p = X_arr.shape
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=device).unsqueeze(0)

        model = cMLP(
            num_series=p,
            lag=int(lag),
            hidden=[int(hidden)],
            activation='relu'
        ).to(device)

        _ = train_model_adam(
            model, X_t,
            lr=float(lr),
            max_iter=int(max_iter),
            lam=float(lam),
            lam_ridge=float(lam_r),
            penalty='GL',
            check_every=int(check_every),
            lookback=5,
            verbose=0
        )

        with torch.no_grad():
            S_run = model.GC(threshold=False).detach().cpu().numpy()

        S_list.append(S_run)

    S_avg = np.mean(np.stack(S_list, axis=0), axis=0)

    return best, results, S_avg


## SAVE LEARNED CAUSAL MATRIX
def _to_jsonable(obj):
    """Recursively convert NumPy / torch scalar types to Python native types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_ngc_results(
    save_dir,
    S,
    feature_names=None,
    hparams: dict | None = None,
    GC_bin=None,
    edge_index=None,
    edge_weight=None,
    results_filename="ngc_results_final.npz",
    meta_filename="ngc_meta_final.json"
):
    os.makedirs(save_dir, exist_ok=True)

    save_dict = {
        "S": np.asarray(S) if S is not None else np.array([]),
        "GC_bin": np.asarray(GC_bin) if GC_bin is not None else np.array([]),
    }

    if edge_index is not None:
        save_dict["edge_index"] = (
            edge_index.detach().cpu().numpy()
            if isinstance(edge_index, torch.Tensor)
            else np.asarray(edge_index)
        )

    if edge_weight is not None:
        save_dict["edge_weight"] = (
            edge_weight.detach().cpu().numpy()
            if isinstance(edge_weight, torch.Tensor)
            else np.asarray(edge_weight)
        )

    np.savez(os.path.join(save_dir, results_filename), **save_dict)

    meta = {
        "feature_names": list(feature_names) if feature_names is not None else None,
        "hparams": hparams or {}
    }

    meta = _to_jsonable(meta)

    with open(os.path.join(save_dir, meta_filename), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved to {save_dir}")
    

## LOAD THE LEARNED CAUSAL MATRIX

def load_ngc_results(
    save_dir,
    device=None,
    results_filename="ngc_results_final.npz",
    meta_filename="ngc_meta_final.json"):
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(os.path.join(save_dir, results_filename), allow_pickle=True)

    S = data["S"] if "S" in data.files and data["S"].size > 0 else None
    GC_bin = data["GC_bin"] if "GC_bin" in data.files and data["GC_bin"].size > 0 else None

    edge_index = None
    if "edge_index" in data.files and data["edge_index"].size > 0:
        edge_index = torch.tensor(data["edge_index"], dtype=torch.long, device=device)

    edge_weight = None
    if "edge_weight" in data.files and data["edge_weight"].size > 0:
        edge_weight = torch.tensor(data["edge_weight"], dtype=torch.float32, device=device)

    with open(os.path.join(save_dir, meta_filename), "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta.get("feature_names", None)
    hparams = meta.get("hparams", {})

    return S, GC_bin, edge_index, edge_weight, feature_names, hparams



## CAUSAL GRAPH EXTRACTION (row-wise quantile)
def build_graph_row_quantile(
    S,
    row_q=0.90,
    remove_self=True,
    abs_threshold=None,
    ensure_one_edge=False
):
    S = S.copy()
    n = S.shape[0]
    edges = []

    for i in range(n):
        row = S[i].copy()

        if remove_self:
            row[i] = -np.inf

        finite_vals = row[np.isfinite(row)]
        if len(finite_vals) == 0:
            continue

        thr = np.quantile(finite_vals, row_q)

        selected = []
        for j in range(n):
            if not np.isfinite(row[j]):
                continue
            if row[j] >= thr:
                if abs_threshold is None or row[j] >= abs_threshold:
                    selected.append((j, row[j]))

        if ensure_one_edge and len(selected) == 0:
            j = np.argmax(row)
            if np.isfinite(row[j]):
                selected.append((j, row[j]))

        for j, score in selected:
            edges.append((j, i, float(score)))
            
    edge_index = np.array([[u, v] for (u, v, _) in edges], dtype=np.int64).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edges, edge_index


def adding_self_loop(S, edge_index):
    num_nodes  = S.shape[0]
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    return edge_index

## SAVE CAUSAL GRAPH
def save_graph(base_dir, edge_index, folder_name, graph_name, feature_map, extra_meta=None):
    folder = base_dir / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    edge_index = edge_index.long().contiguous()
    torch.save(edge_index.cpu(), folder / "edge_index.pt")

    meta = {
        "graph_name": graph_name,
        "num_nodes": len(feature_map),
        "num_edges": int(edge_index.shape[1]),
        "directed": True,
        "self_loops_included": True
    }

    if extra_meta:
        meta.update(extra_meta)

    with open(folder / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved: {folder}")
