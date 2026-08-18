# Engineering tasks

Source: trimmed from [`BUILD.md`](../BUILD.md). Formal research requirements in
[`RESEARCH_CONTRACT.md`](../RESEARCH_CONTRACT.md) take priority.

## Outcome

Provide tested, efficient model code and a model-neutral training harness so ML
work is not blocked by infrastructure.

## Model work

- Maintain the paper-aligned continuous BDH core with input shape
  `[batch, time, d_model]`.
- Preserve strict causal reads, persistent state, per-stream reset, masking,
  chunk continuation, and state detachment.
- Keep a short-sequence parallel implementation as a correctness check only.
- Implement BDH-L according to [`bdh_l_spec.md`](bdh_l_spec.md): one MEG input
  projection followed by the unchanged BDH core.
- Implement BDH-S according to [`bdh_s_spec.md`](bdh_s_spec.md): separate value
  and sparse input paths, with only the first BDH layer using direct sparse
  activation.
- Do not add a separate MEG encoder or task-specific prediction head to the BDH
  core.
- Support small, medium, and larger configurations without hard-coding the
  number or order of MEG channels.

## Training and evaluation infrastructure

- Replace the placeholder trainer with a configuration-driven loop for
  training, validation, checkpointing, reload, and prediction export.
- Save model, optimizer, scheduler, random-number-generator, configuration,
  dataset hash, split hash, preprocessing hash, and recurrent state policy in
  every checkpoint.
- Support mixed precision, gradient clipping, truncated backpropagation, and
  resumable runs.
- Provide adapters for the external PNPL CNN and the project baselines without
  copying their model logic into BDH.
- Record parameter count, examples per second, peak GPU memory, latency, and
  state size for every run.
- Write prediction files keyed by event ID so evaluation can run without
  retraining.

## Tests

- Forward and backward pass on synthetic input.
- Full-sequence output matches chunked execution with dropout disabled.
- Invalid samples do not read or update state.
- Resetting one batch stream does not affect another.
- Checkpoint and reload reproduce predictions.
- BDH-L gradients reach its MEG projection and BDH core.
- BDH-S gradients reach both input paths and the BDH core.
- Recurrent output continues to match the paper implementation at the supported
  reference settings.

## Performance work

- Profile before scaling beyond small models.
- Remove the Python timestep bottleneck using compilation, a scan, or a fused
  kernel while retaining the correctness oracle.
- Avoid unstructured sparse GPU operations unless benchmarks show a real gain.
- Benchmark state memory before approving `n_neurons=32768`.
- Keep analysis activation capture optional so normal training does not retain
  the complete sparse sequence.

## Handoffs

Engineering provides:

1. synthetic-input model tests;
2. one successful real-batch training step;
3. a checkpoint/reload demonstration;
4. profiling results for each approved model size; and
5. stable interfaces consumed by Data and ML/Research.

Engineering is done for formal runs when every required model passes its tests,
overfits the frozen 100-event set through the shared trainer, and can write
reproducible checkpoints and predictions.
