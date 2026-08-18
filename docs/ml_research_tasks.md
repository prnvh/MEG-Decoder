# ML and research tasks

Source: trimmed from `[BUILD.md](../BUILD.md)`. Formal experiments must follow
`[RESEARCH_CONTRACT.md](../RESEARCH_CONTRACT.md)`.

## Outcome

Decide whether BDH provides useful MEG memory or interpretability under fair,
reproducible comparisons. Keep development findings separate from formal
results.

## Before Dataset v1

- Lock the primary task, causal timing, prediction delay, metrics, and stopping
rules.
- Define the loss, class balancing, optimizer budget, model-size budget, and
equal hyperparameter-search budget.
- Validate preprocessing and label mapping with the external PNPL CNN.
- Define parameter-matched linear, GRU, CNN, CNN-recurrent, and causal
Transformer baselines required by the research contract.
- Prepare experiment IDs, configurations, seed handling, prediction formats,
and the immutable run registry.
- Use Dataset v0 to overfit 100 events and debug every model family.
- Treat Dataset v0 scores as development diagnostics only.



## Primary model sequence

1. **External CNN reference:** reproduce inference using its exact expected
  input. Use it to validate the pipeline, not as a formally retrained control.
2. **CNN plus BDH:** where used, feed pre-classifier CNN features into BDH; do
  not feed final class logits into memory.
3. **BDH-L:** test direct latent entry without a secondary learned encoder,
  following `[bdh_l_spec.md](bdh_l_spec.md)`.
4. **BDH-S:** test direct sparse-neuron entry only after its input and first
  layer follow `[bdh_s_spec.md](bdh_s_spec.md)`.
5. **Activation analysis:** build the 2D analysis from stable BDH-S checkpoints,
  not from arbitrary neuron index order.



## Formal experiments

- Use identical target events, preprocessing, splits, context, step budget, and
evaluation code across compared models.
- Keep total trainable parameters within the research contract's tolerance.
- Run formal seeds `{0, 1, 2, 3, 4}`.
- Select models and checkpoints using validation macro-F1 only.
- Open the test split only after model, lag, context, and checkpoint rules are
frozen.
- Report macro-F1, uncertainty, parameter count, FLOPs, peak memory,
examples/second, and latency.
- Save event-keyed predictions and one immutable registry row per run.
- Run 2-second and 30-second primary comparisons before exploratory context,
scale, semantic, or topographic studies.



## Interpretability

- Record sparse activation rates and state norms during training.
- For stable BDH-S checkpoints, measure neuron activation histories, phoneme
selectivity, effective sensor weights, and prediction changes under ablation.
- Construct 2D positions from activation or sensor-weight similarity, then test
stability across sessions and seeds.
- Do not describe a neuron map as anatomical without source localization.
- Do not treat selectivity as causal evidence without neuron or state ablation.



## Continuous evaluation

Every completed checkpoint should automatically produce:

- validation metrics;
- compute and memory profiles;
- context-specific metrics;
- predictions keyed by event ID;
- failure status; and
- a registry entry tied to code, data, split, preprocessing, and config hashes.

