# DNN Assignment 2: RNN, LSTM, GRU and Transformer

Recurrent cells (RNN, LSTM, GRU) and a Transformer encoder, all written from scratch in PyTorch and tested against the library layers. Problem 1 compares the recurrent cells on human-activity recognition (HAR) and a long-range adding problem. Problem 2 builds attention, multi-head attention and an encoder block, then uses them on HAR and on a character-level language model.

| | |
|---|---|
| **Student** | RAKSHANTHA M |
| **Student ID** | 2026AK05368 |
| **Chosen values** | hidden size **h = 64** (RNN, LSTM, GRU) · LM width **d = 128** (4 heads; Task A encoder d = 64) · seeds **0, 1, 2** |
| **5(c) ablation** | ID ends in 8 (even), so **gradient clipping** (norm 1.0 vs none) for the vanilla RNN at T = 200 |
| **Hardware** | Google Colab, T4 GPU |
| **Run time on a T4** | Problem 1: 17 min · Problem 2: 20 min (includes the optional bonus) · **Total: 37 min** |

> As instructed, the variants table is not used: I chose h = 64, d = 128 and seeds 0, 1, 2. With these values the parameter counts equal the ones quoted in the brief (RNN 4,736 · LSTM 18,944 · GRU 14,208 for d = 9; 196,608 block weights for d = 128, d_ff = 512).

---

## Results at a glance

All numbers are mean ± std over 3 seeds. The full tables and figures are in `problem1/figures/` and `problem2/figures/`.

**HAR test accuracy (Problem 1, h = 64)**

| Model | Accuracy | Parameters |
|---|---|---:|
| RNN | 0.810 ± 0.011 | 5,190 |
| LSTM | 0.897 ± 0.008 | 19,590 |
| GRU | 0.900 ± 0.007 | 14,790 |

**Adding problem: runs solved (MSE < 0.05) out of 3**

| Model | T = 50 | T = 100 | T = 200 | T = 400 |
|---|:---:|:---:|:---:|:---:|
| RNN | 1/3 | 0/3 | 0/3 | 0/3 |
| LSTM | 3/3 | 2/3 | 0/3 | 0/3 |
| GRU | 3/3 | 3/3 | 3/3 | 3/3 |

**Problem 2**

| Experiment | Result |
|---|---|
| Task A: Transformer encoder on HAR | no positions 0.926 ± 0.011 · learned 0.914 ± 0.011 · sinusoidal 0.903 ± 0.010 (GRU: 0.900) |
| Bonus: character LM, 5,000 iterations | LSTM 1.542 ± 0.007 nats (45 s, 280,897 params) vs Transformer 1.563 ± 0.006 nats (198 s, 810,049 params) |

---

## Quick start

1. **Problem 1:** open `problem1/problem1.ipynb` and run *Restart and Run All*.
2. **Problem 2:** open `problem2/problem2.ipynb` and run *Restart and Run All* **after Problem 1**, because Task A reads the best Problem 1 model from `results.csv`.

On Colab, upload this folder (or put it in Drive) and change directory into the notebook's folder first, for example `%cd <this folder>/problem1`. All paths in the notebooks are relative to the notebook's own folder.

For a local run: `pip install -r requirements.txt` (the file lists the exact versions; on a machine without the `2.14.1+cu130` CUDA build, install a matching PyTorch first).

| Notebook | Downloads | Writes |
|---|---|---|
| `problem1.ipynb` | UCI HAR into `problem1/data/` | figures in `problem1/figures/`, Problem 1 rows in `../results.csv` |
| `problem2.ipynb` | Tiny Shakespeare into `problem2/data/`; reuses the HAR data from Problem 1 (or downloads it) | figures in `problem2/figures/`, `generated_samples.txt`, Problem 2 rows in `../results.csv` |

Re-running a notebook replaces its own earlier rows in `results.csv`. Data folders are not included in the zip. Seeds are fixed (random, numpy, torch); small run-to-run differences on a GPU are possible. Both notebooks were run top to bottom with papermill, from inside each notebook's folder.

---

## Folder contents

```
2026AK05368_DNN_Assignment/
├── README.md
├── requirements.txt
├── report.pdf
├── results.csv                  one row per run, 12 columns, Problem 1 and 2
├── problem1/
│   ├── problem1.ipynb
│   └── figures/                 plots and the table_*.csv files behind the report
└── problem2/
    ├── problem2.ipynb
    ├── generated_samples.txt    500-character samples at temperatures 0.2, 0.8, 1.5
    └── figures/
```

---

## Training settings

| Experiment | Settings |
|---|---|
| HAR (Problem 1 and Task A) | Adam, lr 1e-3, batch 64, 30 epochs, same for all models; test uses the best-validation checkpoint |
| Adding problem | Adam, lr 1e-3, batch 64, 30 epochs, gradient-norm clipping 1.0 for all main runs; T = 50, 100, 200, 400; 10,000 / 1,000 / 1,000 sequences (data seeds 0 / 1 / 2) |
| Hidden-size sweep 5(b) | LSTM and GRU on HAR at h = 16, 32, 64, 128, 3 seeds each |
| Character LM | context 128, d = 128, 4 heads, 4 layers, dropout 0.1, AdamW (peak lr 1e-3, one-cycle), batch 64, 5,000 iterations; pre-LayerNorm encoder blocks with a final LayerNorm. Heads and no-mask ablations: 1,500 iterations each |
| 5(d) scaling | batch 16, one training step of the full model and of my attention function alone, at context 64, 128, 256 and 512. Memory = activations only (weights and optimiser state excluded); peak memory needs a CUDA GPU |
| Bonus: LSTM vs Transformer | 2-layer LSTM (h = 128) and the Transformer, 5,000 iterations, 3 seeds, gradient clipping 1.0 for both. The LSTM uses AdamW with a constant lr of 1e-3 and no dropout; the Transformer uses the one-cycle schedule and dropout 0.1, so the comparison is not perfectly matched |

---

## Assumptions

- **GRU verification:** PyTorch's candidate-gate hidden bias `b_hn` sits inside `r ⊙ (W_hn h + b_hn)` and cannot be merged into a single bias, so it is set to zero in the library layer before the comparison.
- **3-token attention example:** no instructor example is provided, so I made my own (Q = K = [[1,0],[0,1],[1,1]], V = [[1,2],[3,4],[5,6]]), computed by hand (report, section 2.1) and by code; the two agree to four decimals.
- **Task A:** patches of 8 time steps (16 tokens + CLS); encoder d = 64, 4 heads, 2 layers.
