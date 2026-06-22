### Evaluation Protocol & Metrics
**Core Metrics:** NDCG, Recall@1, MRR, Inference Time, Inference Memory.
**Experimental Setup Departures (vs. standard EBM paper):**
- **Multiple Ground Truths:** Construct function pools containing multiple valid ground truths (similar functions) rather than a single target.
- **Computation Cost:** Strictly record inference time and memory to evaluate student model efficiency against large function pools.
- **Pool Scalability:** Evaluate and report performance stability across varying function pool sizes.

---

### BinaryCorp 3M Dataset
*Focus: Overall embedding performance and baseline comparisons.*

**Baselines (Pre-trained / Zero-shot on BinaryCorp):**
- [ ] **A1** — Nova (Base)
- [ ] **A2** — CLAP
- [ ] **A3** — JTrans

**Baselines (Fine-tuned):**
- [ ] **A4** — Asm2Vec*
- [ ] **A5** — PalmTree* *(Note: Requires angr data-preprocessing integration; interface to be provided to supervisor)*

**EBM & Distillation Variants:**
- [ ] **A6** — Nova + EBM@2&3 (Stages 2 & 3: MNTP + Contrastive only)
- [ ] **A7** — Nova + EBM (Full pipeline: Causal + MNTP + Contrastive)
- [ ] **A8** — Qwen + EBM@2&3
- [ ] **A9** — Nova + Our Teacher
- [ ] **A10** — Nova + Our Student
- [ ] **A11** — Nova + EBM@1 + Our Student (Causal decoder + Student distillation)

---

### Bench (NewData)
*Focus: Merged dataset (BinaryCorp + Custom). Single EBM trained on the combined dataset to reduce computational overhead. Evaluated on cross-architecture, cross-optimization, cross-obfuscation, and cross-compiler splits.*

**Baselines (InfoNCE Fine-tuned):**
- [x] **B1** — CLAP* 
- [ ] **B2** — JTrans* 
- [ ] **B3** — Asm2Vec* 
- [ ] **B4** — PalmTree* 

**Our Pipeline:**
- [x] **B5** — Nova Teacher (Cross-Attention variant) training and evaluation.
- [x] **B6** — Nova Student distillation (non-CA) and evaluation.
- [ ] **B7** — Nova Student distillation (from CA teacher) and evaluation. *(Epoch 5 fallback)*

**EBM Pipeline:**
- [x] **B8** — Nova + EBM@3-only (Contrastive loss from raw Nova).
- [ ] **B9** — Nova + EBM Full (Stages 1+2+3).

---

### Architectural Ablations
*Objective: Isolate the impact of the translation/causal objective versus optimization strategy.*

- [x] **C1** — EBM Stage 3 only (Contrastive only, no causal language modeling).
- [ ] **C2** — EBM Full (Causal + Contrastive).
- [x] **C3** — CLAP* (InfoNCE fine-tuned for direct contrastive baseline).
- [ ] **C4** — Aggregated comparison answering the decoder/optimization hypothesis.