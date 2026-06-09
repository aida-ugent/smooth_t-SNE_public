# Affinity Row Smoothing in t-SNE: Reproducible Experiments

A power-transform approach to row-wise smoothing of t-SNE conditional probabilities. For each point *i*, the affinity row is transformed as p → p^γ / Z (γ < 1 smooths, γ > 1 sharpens), preserving neighbor rank order and support. This acts as an adaptive-perplexity variant that redistributes probability mass more evenly across neighbors without changing the t-SNE graph structure.

**Paper:** "Affinity Row Smoothing in t-SNE: A Power Transform on Conditional Probabilities"

---

## Quick Start (≈5 minutes)

### Windows (PowerShell)
```powershell
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install --default-timeout=1000 -r requirements.txt

# 3. Run MNIST experiment
python smooth_tsne.py --dataset mnist

# Results saved to: results/mnist_smooth_tsne/
```

### Linux / macOS (Bash)
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install --default-timeout=1000 -r requirements.txt

# 3. Run MNIST experiment
python smooth_tsne.py --dataset mnist

# Results saved to: results/mnist_smooth_tsne/
```

---

## System Requirements

- **Python:** 3.10+
- **OS:** Windows, Linux, or macOS
- **RAM:** 8+ GB recommended (MNIST: 2 GB, Mouse: 8+ GB)
- **Disk:** 5+ GB for all results
- **Internet:** Required for first-run dataset downloads (MNIST, Adult)

---

## Complete Setup Guide

### 1. Clone / Download Repository
```bash
git clone <repo-url>
cd smooth-tsne-paper
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
# Create venv
python -m venv venv

# Activate venv
venv\Scripts\activate

# You should see (venv) in your prompt
```

**Linux / macOS (Bash):**
```bash
# Create venv
python -m venv venv

# Activate venv
source venv/bin/activate

# You should see (venv) in your prompt
```

### 3. Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install from requirements.txt
# Use --default-timeout=1000 for slow connections
pip install --default-timeout=1000 -r requirements.txt
```

**Verify installation:**
```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib, openTSNE; print('✓ All dependencies installed!')"
```

### 4. Verify Setup
```bash
# Test the main script loads correctly
python smooth_tsne.py --help

# Should show usage information without errors
```

---

## Project Structure

```
smooth-tsne-paper/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── smooth_tsne.py                      # Main experiment script (8 experiments)
├── utils_smoothing.py                  # Core utilities & algorithms
├── mouse-data/                         # [Optional] Mouse cortex dataset
│   ├── tasic2018.pickle
│   └── importantGenesTasic2018.npy     # Auto-generated on first run
└── results/                            # Output folder (created automatically)
    ├── mnist_smooth_tsne/              # MNIST results
    ├── mouse_smooth_tsne/              # Mouse results (if run)
    └── adult_smooth_tsne/              # Adult results (if run)
```

---

## Datasets

| Dataset | Size | Availability | Setup |
|---------|------|-------------|-------|
| **MNIST** | ~70 MB | Auto-downloaded | No setup needed |
| **Mouse cortex** (Tasic 2018) | ~900 MB | Manual download | Place `tasic2018.pickle` in `mouse-data/` |
| **Adult (UCI)** | ~2 MB | Auto-downloaded | No setup needed |

### Using Mouse Cortex Data

If you have the mouse cortex dataset:
1. Download `tasic2018.pickle` from the original source
2. Place it in the `mouse-data/` directory
3. The `importantGenesTasic2018.npy` file will be auto-generated

---

## Running Experiments

### MNIST (Recommended First Run)
```bash
# Basic run (5000 samples, all 8 experiments)
python smooth_tsne.py --dataset mnist

# Faster: Skip time-consuming experiments 2, 6
python smooth_tsne.py --dataset mnist --skip_exp2 --skip_exp6

# Faster: MNIST with subsampling
python smooth_tsne.py --dataset mnist --mnist_subsample 2000
```

### Mouse Cortex (Full Dataset)
```bash
# All experiments (requires ~8GB RAM, ~30 minutes per experiment)
python smooth_tsne.py --dataset mouse --out_dir results/mouse_smooth_tsne

# Subsample to 5000 cells for faster iteration
python smooth_tsne.py --dataset mouse --mouse_subsample 5000
```

### Adult Census Data
```bash
python smooth_tsne.py --dataset adult --out_dir results/adult_smooth_tsne
```

### Performance Tuning

**Use more CPU cores:**
```bash
python smooth_tsne.py --dataset mnist --n_jobs 8  # Use 8 cores
```

**Limit to fewer cores:**
```bash
python smooth_tsne.py --dataset mnist --n_jobs 2  # Use 2 cores
```

**Skip specific experiments:**
```bash
python smooth_tsne.py --dataset mnist --skip_exp3 --skip_exp4 --skip_exp8
```

**Regenerate plots only (no recomputation):**
```bash
python smooth_tsne.py --dataset mnist --plot_only
```

---

## Experiments Overview

| # | Name | Time | Output |
|---|------|------|--------|
| 1 | Affinity row sharpness | 2 min | `exp1_affinity_sharpness/` |
| 2 | Effective perplexity (distribution + heatmap) | 30 min | `exp2_eff_perplexity/` |
| 3 | Δ perplexity correlations | 10 min | `exp3_delta_perp/` |
| 4 | Neighborhood Overlap (γ sweep) | 15 min | `exp4_neighborhood_overlap_gamma_sweep/` |
| 5 | Embedding comparison (standard vs smooth) | 10 min | `exp5_embedding/` |
| 6 | Sensitivity heatmaps (AUC) | 30 min | `exp6_sensitivity/` |
| 7 | Neighborhood Overlap (comparison) | 15 min | `exp7_neighborhood_overlap_comparison/` |
| 8 | Global Spearman vs γ | 20 min | `exp8_global_spearman/` |

**Total MNIST runtime:** ~2 hours (full 8 experiments)

Each experiment saves:
- PDF figures in `plot_*.pdf`
- Data tables in `*.csv` (for regeneration with `--plot_only`)

---

## Common Command Examples

```bash
# One-off: Quick sanity check
python smooth_tsne.py --dataset mnist --skip_exp2 --skip_exp6 --mnist_subsample 1000

# Standard: Full MNIST experiments
python smooth_tsne.py --dataset mnist

# Advanced: Custom perplexity & gamma values
python smooth_tsne.py --dataset mnist --perplexity 50 --gamma_s 0.5 --gamma_h 2.0

# Fast iteration: Replot from saved CSVs
python smooth_tsne.py --dataset mnist --plot_only

# Parallel processing: Use all CPU cores
python smooth_tsne.py --dataset mnist --n_jobs -1

# Reproducible: Set random seed
python smooth_tsne.py --dataset mnist --random_state 42
```

---

## Key Parameters

```bash
# Dataset selection (required)
--dataset {mnist, mouse, adult}

# Output & data
--out_dir PATH                  # Output directory (auto-generated if omitted)
--mouse_data_dir PATH           # Mouse dataset location
--mnist_subsample N             # MNIST sample size (default: 5000)
--mouse_subsample N             # Mouse cell count (default: all)
--adult_max_rows N              # Adult rows (default: 5000)

# Algorithm parameters
--perplexity P                  # t-SNE perplexity (default: 30)
--gamma_s GAMMA                 # Smoothing γ (default: 0.7)
--gamma_h GAMMA                 # Sharpening γ (default: 1.5)

# Experiment selection
--skip_exp1 ... --skip_exp8     # Skip specific experiments
--plot_only                     # Regenerate plots from CSVs only

# Performance
--n_jobs N                      # Parallel workers: -1=all, 1=single (default: -1)
--n_iter_ee N                   # Early exaggeration iterations (default: 250)
--n_iter_main N                 # Main optimization iterations (default: 750)

# Reproducibility
--random_state SEED             # Random seed (default: 42)
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'numpy'`
**Solution:** Ensure virtual environment is activated
```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Issue: Installation timeout on slow connection
**Solution:** Use extended timeout
```bash
pip install --default-timeout=1000 -r requirements.txt
```

### Issue: Out of memory errors
**Solution:** Subsample the data
```bash
python smooth_tsne.py --dataset mnist --mnist_subsample 2000
python smooth_tsne.py --dataset mouse --mouse_subsample 5000
```

### Issue: Slow computation
**Solution:** Use fewer cores or skip experiments
```bash
python smooth_tsne.py --dataset mnist --n_jobs 2 --skip_exp2 --skip_exp6
```

### Issue: `FileNotFoundError` for mouse data
**Solution:** Ensure `tasic2018.pickle` is in `mouse-data/` directory
```bash
# Check if file exists
ls mouse-data/tasic2018.pickle    # Linux/macOS
dir mouse-data                     # Windows
```

### Issue: Plots not generating / looking strange
**Solution:** Use `--plot_only` to regenerate from cached CSVs
```bash
python smooth_tsne.py --dataset mnist --plot_only
```

---

## Reproducibility Notes

- All random operations use `--random_state 42` by default for reproducibility
- Results are saved as CSV files alongside PDFs for plot regeneration
- Use `--plot_only` to regenerate all figures without recomputation
- Dependencies are pinned to exact versions in `requirements.txt`
- Python 3.10+ required (tested on 3.10, 3.11, 3.12, 3.13)

---

## Citation

If you use this code, please cite:

```bibtex
@article{smooth-tsne,
  title={Affinity Row Smoothing in t-SNE: A Power Transform on Conditional Probabilities},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

---

## License

[Add your license here]

---

## Support

For issues or questions:
1. Check **Troubleshooting** section above
2. Verify Python version: `python --version` (should be 3.10+)
3. Check virtual environment: You should see `(venv)` in your terminal
4. Test dependencies: `python -c "import numpy, pandas, scipy, sklearn, matplotlib, openTSNE; print('OK')"`
