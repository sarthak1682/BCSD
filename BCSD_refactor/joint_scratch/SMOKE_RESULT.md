# BinaryCorp Bench joint smoke result

This is a superseded plumbing check, not a scientific result. It was run before
the data source was switched from Nova-format bench assembly to CLAP-strict
bench assembly.

- model: 1,045,328 randomly initialized parameters;
- data: 64 training and 16 binary-disjoint validation pairs;
- schedule: one epoch, 32 coupled optimizer updates;
- sequence length: 64 tokens;
- objectives: LM cross-entropy and retrieval contrastive loss every update.

Validation after the smoke run:

| Metric | Value |
|---|---:|
| LM cross-entropy | 10.2617 |
| LM perplexity | 28,614.65 |
| LM token accuracy | 0.0 |
| Retrieval contrastive loss | 1.2535 |
| Global retrieval Recall@1 | 0.125 |

For reference, a uniform 32,280-token model has cross-entropy
`ln(32280) = 10.382`. The movement below that value and finite gradients for
both task tokens confirm that the two objectives are connected and updating the
shared model. Sixteen queries are far too few to interpret Recall@1, and zero
token accuracy is expected after only 32 updates on a one-million-parameter
model.

Artifacts are in
`model_checkpoints/joint_scratch/smoke_bench_small_codex/`.
