# DNN Assignment 2: RNN, LSTM, GRU and Transformer

**RAKSHANTHA M · Student ID 2026AK05368**

## Variant

- **Values used:** hidden size h = 64 (RNN, LSTM, GRU); language-model width d = 128 (4 heads, 4 layers; the Task A encoder uses d = 64); seeds 0, 1, 2. The variants table was not in the released brief, so these values were chosen (assumption).
- **5(c) ablation:** the ID ends in 8 (even), so it is gradient clipping (norm 1.0 vs none) for the vanilla RNN at T = 200.

## Environment

- **Software:** Python 3.10+, PyTorch 2.14.1 (CUDA 13.0 build), NumPy, pandas, Matplotlib, scikit-learn. Exact versions are in `requirements.txt`; install with `pip install -r requirements.txt`. Without that CUDA build, install a matching PyTorch first.
- **Hardware:** Google Colab, T4 GPU.

## How to run

1. Open `problem1/problem1.ipynb` and run *Restart and Run All*. It downloads UCI HAR into `problem1/data/` and writes figures to `problem1/figures/` and Problem 1 rows to `../results.csv`.
2. Then open `problem2/problem2.ipynb` and run *Restart and Run All*. It needs Problem 1 to have run first, because Task A reads the best Problem 1 model from `results.csv`. It downloads Tiny Shakespeare into `problem2/data/`, reuses the HAR data, and writes `generated_samples.txt`, figures and Problem 2 rows.

All paths are relative to each notebook's own folder. On Colab, `%cd` into the notebook's folder first. Re-running a notebook replaces its own rows in `results.csv`. Datasets are not included in the zip.

## Total run time on a T4

- Problem 1: 17 min
- Problem 2: 20 min (including the bonus)
- **Total: 37 min**

## Assumptions and known issues

- **GRU check:** PyTorch's `b_hn` sits inside `r ◦ (W_hn h + b_hn)`, but my GRU cell keeps one merged bias per gate. So `b_hn` is set to zero in the library layer before the comparison.
- **3-token example:** the instructor example was not supplied, so I made my own (report, section 2.1). The hand and code results agree to four decimals.
- **Bonus comparison:** the LSTM uses a constant lr of 1e-3 and no dropout. The Transformer uses a one-cycle schedule and dropout 0.1, so the comparison is not perfectly matched.
- **No-mask sample:** `problem2/figures/nomask_sample.txt` is almost all newlines. This is the expected result of removing the causal mask, not a bug.
- **Reproducibility:** seeds are fixed (random, numpy, torch), but small run-to-run differences on a GPU are possible.
- Everything else runs top to bottom without errors. Nothing is known not to work.
