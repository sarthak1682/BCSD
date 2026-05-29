# Nova EBM Benchmarking Pipeline

Minimalist guide to setup and execute the 3-stage training pipeline (Cross-opt translation, MNTP, and Contrastive learning).

## 1. Setup Environment
Installs all dependencies (with pinned working versions), fetches the Hugging Face `lt-asset/nova-1.3b` model snapshot, and retrieves the rebalanced benchmark datasets:

```bash
bash BCSD_refactor/setup_env.sh
```

## 2. Run Pipeline

### Option A: Server Cluster (SLURM)
To submit the 3-stage training pipeline to the cluster queue (runs in the background on an allocated GPU node):

```bash
sbatch BCSD_refactor/slurm_scripts/nova_ebm_bench.sbatch
```

**Before submitting**, open `BCSD_refactor/slurm_scripts/nova_ebm_bench.sbatch` and update one line:
- `#SBATCH --partition=gpu` → replace `gpu` with your cluster's partition name (run `sinfo` to see available partitions)

### Option B: Interactive Node / Local (Plain Bash)
To run the training pipeline directly in your current terminal session:

```bash
bash BCSD_refactor/run_pipeline.sh
```

