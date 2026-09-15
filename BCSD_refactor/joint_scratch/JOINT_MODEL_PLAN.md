# Joint embedding + generation model plan

## Current task switching

The current `joint_scratch` model does not infer the task from the input text alone. The training loop explicitly chooses the task, then calls a task-specific forward path:

- `forward_lm(...)` for causal translation / generation
- `forward_retrieval(...)` for embedding / BCSD retrieval

Inside the shared model, both paths use the same token embedding and transformer layers, but prepend a learned task embedding before the sequence:

- `Task.LM` gets task embedding index `0`
- `Task.RETRIEVAL` gets task embedding index `1`

The attention contract also changes by task:

- LM task: strict causal attention mask, tied LM head, shifted cross-entropy over target tokens.
- Retrieval task: full bidirectional attention, attention pooling, projection head, contrastive loss.

With:

```bash
--training-mode alternate --lm-task-prob 0.6
```

training samples one task per batch:

- about 60% LM / causal translation batches
- about 40% retrieval / contrastive batches

This is not MNTP/MLM yet. The retrieval path is bidirectional, but it is supervised by contrastive labels, not masked token prediction.

## What we are trying to prove

The near-term claim is not "we have a good generation model already." The near-term claim is:

> A single shared model can learn useful BCSD embeddings while retaining a real causal generation objective.

The first hard retrieval target is jTrans on the local BinaryCorp bench variant, not the BinaryCorp-3M table from the EBM paper.

## Local BinaryCorp bench targets

Local benchmark protocol:

- test data: `/home/ra72yeq/projects/NovaXLLM2Vec/nvemb/output_benchset_rebalanced_test_*_strict.jsonl`
- queries: 45,000
- unique target functions: 4,988
- evaluator: `binarycorp_bench/eval_bench.py` / `joint_scratch.evaluate --protocol bench`

Current local reports:

| model/report | Pool50 R@1 | Pool100 R@1 | Pool200 R@1 | Pool500 R@1 | full-bank R@1 | full-bank R@5 | full-bank R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| jTrans stronger local report | 0.5492 | 0.4849 | 0.4524 | 0.3898 | 0.4250 | 0.5330 | 0.5611 |
| jTrans local finetuned-best | 0.5566 | 0.4793 | 0.4205 | 0.3474 | 0.3675 | 0.4844 | 0.5227 |
| current 30M joint alternate p60 | 0.6104 | 0.5422 | 0.4789 | 0.4013 | 0.2771 | 0.3787 | 0.4272 |

Do not use the earlier local CLAP-finetuned full-bank `0.600` R@1 number as
evidence; that report is treated as buggy/excluded. Keep CLAP comparisons to
verified reruns only.

Interpretation:

- Current 30M joint model already beats jTrans on pooled recall.
- It does not beat jTrans on full-bank retrieval.
- The first serious target is therefore:

```text
full-bank Recall@1 > 0.42498
```

## Architecture target for jTrans-sized comparison

jTrans is treated as the retrieval-specialist baseline at about 88M parameters.

The closest clean joint-scratch size using our existing architecture is:

```text
d_model=768
layers=8
heads=12
ff_dim=3072
embedding_dim=768
max_length=512 or 1024
```

Estimated parameters:

| config | parameters |
|---|---:|
| d=768, 8 layers, max_length=512 | 83.1M |
| d=768, 8 layers, max_length=1024 | 83.5M |
| d=768, 10 layers, max_length=512 | 97.2M |

Recommendation: use the 83M variant first. It is close enough to 88M and avoids paying for a larger model before we know the training recipe is right.

## Generation baseline at similar size

There is no perfect public "88M binary assembly generation" baseline. For fair comparison, use two generation baselines:

1. **Same-architecture causal-only baseline**
   - Same tokenizer, same BinaryCorp bench CLAP-strict train pairs, same 83M transformer.
   - Train only `forward_lm`.
   - This is the cleanest specialized generation baseline because it controls architecture, tokenizer, and data.

2. **External code-generation reference**
   - `Salesforce/codet5-small` is a reasonable public small code-generation reference.
   - It is encoder-decoder, around the small-code-model regime, and its model card lists code generation, code translation, code refinement, and clone detection as intended downstream tasks.
   - It is not binary-assembly-specialized, so it should be treated as context, not the main fairness anchor.

`EleutherAI/gpt-neo-125m` is closer above 88M as a causal LM, but it is general text generation rather than code/binary generation. I would not use it as the main generation baseline unless we specifically want "small causal LM adapted to asm."

## Dataset suitability

The CLAP-strict BinaryCorp bench variant is compatible with the joint idea because it has:

- multiple variants per function;
- compiler / optimization / architecture / obfuscation metadata folded into `opt`;
- train/test split suitable for retrieval evaluation;
- raw assembly without Nova `<label-N>` helper tokens.

But it has an important generation limitation:

- It gives a target assembly string for causal translation, so CE/perplexity/token accuracy are valid training diagnostics.
- It does not by itself prove generated assembly is semantically correct, because many target assemblies can be valid for the same function.

So generation evaluation should be reported in layers:

1. teacher-forced LM metrics: CE, perplexity, token accuracy;
2. generation text metrics: exact/prefix/token-level match against the chosen target;
3. retrieval-after-generation: generated target should retrieve the same function;
4. semantic/executable validation only if we can assemble/run or otherwise verify equivalence.

Until layer 3 or 4 exists, generation claims should stay modest.

## Training plan

### Phase 0: current 30M full-data check

Purpose: test whether the current best balance scales with data.

- model: 30M
- data: full CLAP-strict BinaryCorp bench train split
- schedule: alternate batches
- balance: `lm_task_prob=0.6`
- eval: official `--protocol bench`

Decision:

- If full-bank R@1 jumps substantially, scale recipe to 83M.
- If pooled improves but full-bank R@1 stays low, fix retrieval sampling/loss before scaling.

### Phase 0.5: 23.5M causal + MLM pretrain, then contrastive fine-tune

Purpose: test the more defensible joint-distribution recipe before scaling.  What we tried first was jointly pretraining causal translation (decoder mode) and MLM (embedder mode), then fine-tuning the same encoder path with contrastive retrieval.

```text
causal translation + bidirectional MLM pretraining
then
contrastive retrieval fine-tuning
```

This better matches the theory-motivated joint modeling story:

- causal LM models the data/generation side;
- bidirectional MLM teaches the encoder path dense assembly syntax/semantics;
- contrastive fine-tuning turns the encoder representation into a BCSD retrieval space.

Implementation in `joint_scratch`:

- `--training-objectives lm_mlm` alternates causal translation and MLM batches.
- MLM uses the bidirectional/retrieval task token, intentionally pretraining the same encoder path later used for retrieval.
- `--training-objectives retrieval --init-checkpoint .../final` fine-tunes only the contrastive retrieval objective.
- `--no-validation` means training is train-only; benchmark testing is done afterward by `joint_scratch.evaluate --protocol bench`.

Initial schedule:

```text
Stage A/B pretrain:
  model = 23.5M
  objective set = lm_mlm
  training mode = alternate
  lm_task_prob = 0.5
  epochs = 10

Stage C fine-tune:
  init = pretrain/final
  objective set = retrieval
  epochs = 1

Evaluation:
  official BinaryCorp bench protocol on test split
```

Basic 23.5M result:

- LM+MLM10 -> CL1 reaches Pool@500 R@1 = 0.778 and full-bank R@1/R@5/R@10 = 0.517/0.737/0.803 on the official BinaryCorp bench test split.
- The same checkpoint gives optimization-classification macro-F1 = 0.515 for 4-way `O0/O1/O2/O3`, and O0-vs-O3 AUC = 0.980.
- Extending pretraining to LM+MLM15 gives only small retrieval changes: Pool@500 R@1 = 0.783, full-bank R@1/R@5/R@10 = 0.517/0.742/0.806.

Most comparable baselines:

- Conceptually, the closest comparison is EBM-style staged training: use generation/token-prediction to shape the representation, then contrastive fine-tune for retrieval.  Our current version replaces EBM's Nova/Qwen initialization with a scratch 23.5M transformer and uses causal translation + MLM before CL.
- As a retrieval benchmark, compare directly to CLAP, jTrans, EBM-full/no-causal, the distilled Student variants, and asm2vec on the same BinaryCorp bench protocol:

| Model | Pool@500 R@1 | Full R@1 | Full R@5 | Full R@10 |
|---|---:|---:|---:|---:|
| EBM-full | 0.808 | 0.559 | 0.778 | 0.834 |
| EBM-no-causal | 0.764 | 0.526 | 0.726 | 0.785 |
| Student EBM-full | 0.748 | 0.519 | 0.711 | 0.771 |
| Student | 0.737 | 0.514 | 0.702 | 0.758 |
| CLAP | 0.691 | 0.493 | 0.663 | 0.721 |
| jTrans | 0.390 | 0.425 | 0.533 | 0.561 |
| asm2vec | 0.184 | 0.178 | 0.229 | 0.256 |
| Joint scratch 23.5M, LM+MLM10 -> CL1 | 0.778 | 0.517 | 0.737 | 0.803 |
| Joint scratch 83M, LM+MLM10 -> CL1 | 0.794 | 0.530 | 0.760 | 0.823 |

So the 23.5M joint scratch model beats CLAP, jTrans, asm2vec, Student, and Student EBM-full on pooled retrieval and top-5/top-10 full-corpus retrieval; it is just below EBM-full and roughly tied with EBM-no-causal/Student on full R@1.  Scaling to 83M improves every retrieval cutoff, beats EBM-no-causal on full R@1/R@5/R@10, and approaches EBM-full while still using a scratch model.  For optimization classification, jTrans and EBM-full are still stronger on 4-way macro-F1, while joint scratch is close on O0-vs-O3 AUC.

Decision:

- If full-bank R@1 approaches or beats jTrans, scale this staged recipe to 83M.
- If pooled improves but full-bank/cross-arch lags, keep the staged recipe but add cross-arch positive sampling and GTE/cGTE.
- If MLM improves LM metrics but not retrieval, the bottleneck is likely the contrastive sampling/loss, not the joint pretraining idea.

### Phase 1: 83M jTrans-sized joint run

Purpose: compare a joint model to an 88M retrieval specialist.

Config:

```text
d_model=768
layers=8
heads=12
ff_dim=3072
embedding_dim=768
max_length=512 initially
```

Completed run:

```text
pretrain:
  objective set = lm_mlm
  training mode = alternate
  lm_task_prob = 0.5
  epochs = 10

fine-tune:
  objective set = retrieval
  epochs = 1

batch_size=64
```

Retrieval result:

```text
Pool@500 R@1 = 0.794
full-bank R@1/R@5/R@10 = 0.530/0.760/0.823
```

Generation success criterion:

```text
close to or better than causal-only 83M on validation CE / perplexity / token accuracy,
then validate with generated-output metrics.
```

### Phase 2: add retrieval-specific fixes if full-bank R@1 lags

Most likely issue from current p60 result:

```text
same-architecture retrieval is okay, cross-architecture retrieval is weak.
```

Fixes:

1. Cross-arch positive sampling:
   - explicitly sample source/target pairs from different architectures when possible;
   - keep some same-arch pairs so we do not destroy pooled/same-arch recall.

2. Add setting tokens to retrieval mode:
   - architecture/compiler/optimization/obfuscation metadata should be visible to the encoder;
   - EBM uses language/setting tokens for this reason.

3. Replace simple positive-aware InfoNCE with GTE/cGTE-style loss:
   - include q→k, k→q, q→q, k→k terms;
   - optionally accumulate embeddings across microbatches for more negatives.

### Phase 3: strengthen MLM/contrastive if needed

The EBM paper stages causal translation, then MNTP/LLM2Vec, then contrastive/cGTE.

The current MLM implementation is a direct bidirectional masked-token objective over assembly tokens. If retrieval still underperforms after the 30M staged experiment, strengthen the EBM-like pieces:

- use MNTP-style masking/spans rather than simple independent token masks;
- use cross-arch positive sampling before contrastive fine-tune;
- replace simple positive-aware InfoNCE with GTE/cGTE;
- optionally interleave a small amount of LM/MLM during contrastive fine-tuning to preserve the joint model behavior.

## What would count as "beating both"

For the strong claim, the joint model should beat or match two specialists:

1. Retrieval specialist:
   - beat jTrans on local BinaryCorp bench full-bank R@1.

2. Generation specialist:
   - beat or match a causal-only model of the same size trained on the same causal translation data.

The external CodeT5-small comparison is useful context, but the same-architecture causal-only baseline is the fair generation specialist.

## References

- EBM paper: https://dmas.lab.mcgill.ca/fung/pub/LSDFC25neurips.pdf
- EBM code: https://github.com/L1NNA/binary_sim
- CodeT5-small model card: https://huggingface.co/Salesforce/codet5-small
- GPT-Neo-125M model card: https://huggingface.co/EleutherAI/gpt-neo-125m
- Local jTrans stronger report: `binarycorp_bench/jtrans/eval_jtrans_bench_full_report.json`
- Local current joint report: `model_checkpoints/joint_scratch/clap_strict_joint_30m_alternate_p60_bs64_seed42/eval_official_test_best_bench.json`
