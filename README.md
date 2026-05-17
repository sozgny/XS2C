# XS2C — Explainable Sensor-to-Cause Diagnosis

XS2C is a framework for anomaly detection and root cause diagnosis in multivariate time series, based on causal graph neural networks. It consists of two components: **detection** and **diagnosis**.

---

## Repository Structure

| File | Description |
|---|---|
| `all_functions.py` | All model definitions, training, testing, and evaluation functions |
| `xs2c_detection.ipynb` | Detection component — trains the GNN model and evaluates anomaly detection performance |
| `xs2c_diagnosis.ipynb` | Diagnosis component — identifies faulty sensors and ranks root cause hypotheses |
| `swat_best_model.pt` | Pretrained model weights for the SWaT dataset |
| `swat_best_params.json` | Best hyperparameters found for the SWaT dataset |
| `swat_edge_index.pt` | Learned causal graph (edge index) for the SWaT dataset |
| `tep_best_model.pt` | Pretrained model weights for the TEP dataset |
| `tep_best_params.json` | Best hyperparameters found for the TEP dataset |
| `tep_edge_index.pt` | Learned causal graph (edge index) for the TEP dataset |

---

## Data Format

### Train & Test CSV files

Both train and test files must be CSV files with the following structure:

- **Index column**: integer timestamps starting from 0 (0, 1, 2, ...)
- **Feature columns**: one column per sensor/variable
- **Attack column** (test file only): binary column named `attack` with values 0 (normal) or 1 (anomaly)

Example (`test.csv`):

```
,sensor_1,sensor_2,...,sensor_n,attack
0,0.12,0.34,...,0.56,0
1,0.13,0.35,...,0.57,0
2,0.89,0.91,...,0.78,1
...
```

The train file does not need an `attack` column. If present, it will be automatically dropped.

### Variable list (`list.txt`)

A plain text file with one variable/sensor name per line, in the same order as the CSV columns:

```
sensor_1
sensor_2
...
sensor_n
```

### Causal graph (`edge_index.pt`)

A PyTorch tensor of shape `(2, E)` saved with `torch.save()`, where `E` is the number of directed edges. Each column `[:, i]` represents a directed edge from node `edge_index[0, i]` to node `edge_index[1, i]`. Node indices correspond to the order of variables in `list.txt`.

---

## Pretrained Models

Pretrained model weights are provided for both datasets for exact reproduction of paper results. To use them, set `load_model_path` to the corresponding `.pt` file path in the notebook configuration.

| Dataset | Model file | Params file |
|---|---|---|
| SWaT | `swat_best_model.pt` | `swat_best_params.json` |
| TEP | `tep_best_model.pt` | `tep_best_params.json` |

> **Note on reproducibility:** Detection performance metrics are stable across runs with `random_seed=42`. However, due to non-determinism in PyTorch's data loading, retraining from scratch may yield slightly different symptom identification results. For exact paper results, use the provided pretrained weights.

---

## Requirements

- Python 3.10
- PyTorch 2.1.0
- PyTorch Geometric 2.4.0
- pandas, numpy, scikit-learn, matplotlib, seaborn, networkx, pysat

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{yourpaper2025,
  title   = {Your Paper Title},
  author  = {Your Name},
  journal = {Journal/Conference},
  year    = {2025}
}
```
