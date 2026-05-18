# XS2C — eXplainable Sensor-to-Component Framework

XS2C is a framework for anomaly detection and root cause diagnosis in multivariate time series combining attention-based GNN and consistency-based diagnosis. After graph construction, it consists of two components: **detection** and **diagnosis**.

---

## Repository Structure

| File | Description |
|---|---|
| `all_functions.py` | All model definitions, training, testing, and evaluation functions |
| `xs2c_detection.ipynb` | Detection component — trains the GNN model, evaluates anomaly detection performance, then fine-tunes a fault model |
| `xs2c_diagnosis.ipynb` | Diagnosis component — using identified symptoms, generates ranked diagnoses of faulty components |
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

- **Timestamp column**: integer timestamps starting from 0 (0, 1, 2, ...)
- **Feature columns**: one column per sensor/variable
- **Attack column** (test file only): binary column named `attack` with values 0 (normal) or 1 (anomaly)

Example (`test.csv`):

```
timestamp,sensor_1,sensor_2,...,sensor_n,attack
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
## Datasets

### SWaT (Secure Water Treatment)
The SWaT dataset is not publicly available and must be requested from the iTrust research centre at the Singapore University of Technology and Design (SUTD):
- Request access: [https://itrust.sutd.edu.sg/](https://itrust.sutd.edu.sg/)

### TEP (Tennessee Eastman Process)
The TEP dataset was generated using the simulator based on the original Fortran code by J.J. Downs and E.F. Vogel (1993), with modifications by E.L. Russell, L.H. Chiang, and R.D. Braatz:
- Simulator: [https://github.com/jkitchin/tennessee-eastman-profbraatz](https://github.com/jkitchin/tennessee-eastman-profbraatz)

Please cite the original TEP works if you use this simulator:

```bibtex
@article{downs1993plant,
  title={A plant-wide industrial process control problem},
  author={Downs, James J and Vogel, Ernest F},
  journal={Computers \& Chemical Engineering},
  volume={17},
  number={3},
  pages={245--255},
  year={1993}
}

@book{russell2000data,
  title={Data-driven Techniques for Fault Detection and Diagnosis in Chemical Processes},
  author={Russell, Evan L and Chiang, Leo H and Braatz, Richard D},
  publisher={Springer-Verlag},
  year={2000}
}
```

## Requirements

- Python 3.10
- PyTorch 2.1.0
- PyTorch Geometric 2.4.0
- pandas, numpy, scikit-learn, matplotlib, seaborn, networkx, pysat

---

## Acknowledgements

This work builds upon the attention-based graph neural network forecasting architecture from **GDN** (Graph Neural Network-Based Anomaly Detection in Multivariate Time Series, AAAI 2021). Several components in `all_functions.py` are adapted from the original GDN implementation.

- Paper: [https://arxiv.org/pdf/2106.06947.pdf](https://arxiv.org/pdf/2106.06947.pdf)
- Code: [https://github.com/d-ailin/GDN](https://github.com/d-ailin/GDN)

If you use this repository, please also cite the original GDN paper:

```bibtex
@inproceedings{deng2021graph,
  title={Graph neural network-based anomaly detection in multivariate time series},
  author={Deng, Ailin and Hooi, Bryan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4027--4035},
  year={2021}
}
```

---

## Citation

