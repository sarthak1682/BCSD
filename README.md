# Nova EBM Benchmarking Pipeline

Minimalist guide to setup and execute the 3-stage training pipeline (Cross-opt translation, MNTP, and Contrastive learning).

## 1. Setup Environment
Installs all dependencies (with pinned working versions), fetches the Hugging Face `lt-asset/nova-1.3b` model snapshot, and retrieves the rebalanced benchmark datasets:

```bash
bash BCSD_refactor/setup_env.sh
```

## 2. Run Pipeline
Configures the correct `PYTHONPATH` and runs all 3 training stages:

```bash
bash BCSD_refactor/run_pipeline.sh
```
