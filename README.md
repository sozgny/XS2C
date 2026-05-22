# XS2C — eXplainable Sensor-to-Component Framework

XS2C is a framework for anomaly detection and diagnosis in multivariate time series combining an attention-based GNN anomaly detection with theory of consistency-based diagnosis. It consists of three main components: **graph construction**, **GNN-based anomaly detection** and **anomaly diagnosis**.

---

## Repository Structure

```
XS2C/
├── graphs/
│   ├── swat_edge_index.pt        # Learned causal graph for SWaT
│   └── tep_edge_index.pt         # Learned causal graph for TEP
├── notebooks/
│   ├── xs2c_detection.ipynb      # Detection: GNN training & anomaly detection & fine-tuning a faultModel
│   ├── xs2c_diagnosis.ipynb      # Diagnosis: symptom-based diagnosis identification
│   └── graph_construction.ipynb  # Causal graph construction via Neural Granger Causality (NGC)
├── pretrained_models/
│   ├── swat_best_model.pt        # Pretrained weights for SWaT
│   ├── swat_best_params.json     # Best hyperparameters for SWaT
│   ├── tep_best_model.pt         # Pretrained weights for TEP
│   └── tep_best_params.json      # Best hyperparameters for TEP
├── src/
│   ├── all_functions.py          # util, processing, training, evaluation functions, models
│   └── ngc.py                    # NGC implementation
├── TEP_datasets/
│   ├── train.csv                 # 7-day normal operation data
│   └── test_idv*.csv             # Fault scenarios (IDV 1–20)
└── README.md
```
---

## Datasets

### SWaT (Secure Water Treatment)
The SWaT dataset is not publicly available and must be requested from the iTrust research centre at the Singapore University of Technology and Design (SUTD):
- Request access: [https://itrust.sutd.edu.sg/](https://itrust.sutd.edu.sg/)

### TEP (Tennessee Eastman Process)
The TEP dataset was generated using the simulator based on the original Fortran code by J.J. Downs and E.F. Vogel (1993), with modifications by E.L. Russell, L.H. Chiang, and R.D. Braatz:
- Simulator: [https://github.com/jkitchin/tennessee-eastman-profbraatz](https://github.com/jkitchin/tennessee-eastman-profbraatz)
- TEP simulation datasets are included in the `TEP_datasets/` folder.

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

Data is normalized via MinMaxScaler().

### Variable list (`list.txt`)
A plain text file with one variable/sensor name per line, in the same order as the CSV columns:

```
sensor_1
sensor_2
...
sensor_n
```
### Causal Graph Construction
The causal graphs (`edge_index.pt`) were constructed using Neural Granger Causality (NGC). 
The implementation is based on the original NGC code with additional modifications:
- Original code: [https://github.com/iancovert/Neural-GC](https://github.com/iancovert/Neural-GC)
- The original NGC implementation was extended with additional preprocessing, optimization and graph filtering steps for compatibility with XS2C pipeline.

Please also cite the NGC paper if you use the provided causal graphs:

```bibtex
@article{tank2021neural,
  title={Neural granger causality},
  author={Tank, Alex and Covert, Ian and Foti, Nicholas and Shojaie, Ali and Fox, Emily B},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={8},
  pages={4267--4279},
  year={2021}
}
```
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

## Requirements

- Python 3.14.4
- PyTorch 2.11.0
- PyTorch Geometric 2.7.0
- pandas 3.0.2
- numpy 2.4.4
- scikit-learn 1.8.0
- matplotlib 3.10.9
- seaborn 0.13.2
- networkx 3.6.1
- python-sat 1.9.dev2

```bash
pip install torch-geometric pandas numpy scikit-learn matplotlib seaborn networkx
pip install python-sat --pre
```

---

## Acknowledgements

This work builds upon the attention-based graph neural network forecasting architecture from **GDN** (Graph Neural Network-Based Anomaly Detection in Multivariate Time Series, AAAI 2021). Several components in `all_functions.py` are adapted from the original GDN implementation.

- Paper: [https://arxiv.org/pdf/2106.06947](https://arxiv.org/abs/2106.06947)
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

