# DNN Assignment 2: RNN, LSTM, GRU and Transformer

**Student:** RAKSHANTHA M · **Student ID:** 2026AK05368

**Variant:** the last digit of the ID is 8 (even), so the 5(c) ablation is gradient clipping (norm 1.0 vs none) for the vanilla RNN at T = 200. The variants table ("Individualisation") was not included in the released brief, so I assumed hidden size 64, Transformer width d = 128 and seeds 0, 1, 2.

## Environment
Google Colab, T4 GPU (*Runtime → Change runtime type → T4 GPU*), Python 3.13, PyTorch 2.14.1 (CUDA build). `requirements.txt` lists the exact versions.

```
pip install -r requirements.txt
```

## How to run
All paths in the notebooks are relative to each notebook's own folder. On Colab, upload this folder (or put it in Drive) and change directory into the notebook's folder before running, for example `%cd <this folder>/problem1`.

1. `problem1/problem1.ipynb`: run *Restart and Run All*. It downloads UCI HAR into `problem1/data/`, saves figures to `problem1/figures/`, and appends one row per run to `../results.csv`.
2. `problem2/problem2.ipynb`: run *Restart and Run All* **after** Problem 1, because Task A reads the best Problem 1 model from `results.csv`. It reuses the HAR data from Problem 1 (or downloads it) and downloads Tiny Shakespeare into `problem2/data/`. It writes `generated_samples.txt`, saves figures to `problem2/figures/` and appends its rows to `results.csv`.

Data folders are not included in the zip. Seeds are fixed (random, numpy, torch); small run-to-run differences on a GPU are possible.

## Run time on a T4
Problem 1: <fill in> min · Problem 2: <fill in> min · Total: <fill in> min

## Training budget
- HAR (Problem 1 and Task A): Adam, lr 1e-3, batch 64, 30 epochs, same for all models; test uses the best-validation checkpoint.
- Adding problem: Adam, lr 1e-3, batch 64, 30 epochs, gradient-norm clipping 1.0 for all models in the main runs. 10,000 / 1,000 / 1,000 sequences (data seeds 0 / 1 / 2).
- Character LM: context 128, d = 128, 4 heads, 4 layers, dropout 0.1, AdamW (peak lr 1e-3, one-cycle), batch 64, 5,000 iterations. The heads and no-mask ablations use 1,500 iterations each.

## Assumptions
- Variants table missing: hidden size 64, d = 128, seeds 0, 1, 2 (see above).
- GRU verification: PyTorch's candidate-gate hidden bias `b_hn` sits inside `r ⊙ (W_hn h + b_hn)` and cannot be merged into a single bias, so it is set to zero in the library layer before the comparison.
- 3-token attention example: the instructor's example was not supplied, so I used my own (Q = K = [[1,0],[0,1],[1,1]], V = [[1,2],[3,4],[5,6]]), worked out by hand and by code.
- Task A embeds patches of 8 time steps (16 tokens + CLS); encoder d = 64, 4 heads, 2 layers.

## Known issues
- Peak-memory measurement in 5(d) needs a CUDA GPU (on CPU it is reported as not available).
- <anything else that does not work, or "None">
