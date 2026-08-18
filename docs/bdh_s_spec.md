# BDH-S specification

Status: proposed

## Purpose

BDH-S lets each MEG sample directly activate BDH's large sparse neuron space.
It keeps a separate compact value path for the information written into
memory.

The `S` means **sparse-neuron entry**. Its purpose is to test whether direct,
inspectable sensor-to-neuron activation improves long-context learning or
interpretability over BDH-L.

## Data flow

```text
                         -> sparse projection -> sparse neurons [n_neurons]
MEG sample [channels] --|
                         -> value projection  -> latent value [d_model]

sparse neurons + latent value
        -> special first BDH layer
        -> standard remaining BDH layers
        -> hidden output [d_model] + recurrent state
```

For MEG sample `m_t`:

```text
s_t = ReLU(P_sparse(m_t))
v_t = P_value(m_t)
```

In the first layer, `s_t` selects where memory is read and written, while
`v_t` is the content written to memory. RoPE is applied to `s_t` before the
read and update. The layer must read the previous state before writing the
current sample, preserving strict causality.

After the first layer, all later layers use the normal BDH rule that derives
sparse activations from the current latent value.

## Interface

- Input: MEG tensor `[batch, time, channels]`.
- Expected channel count: read from the dataset manifest; do not hard-code 306.
- State, masks, reset behavior, detachment, outputs, and task-head boundaries
  match BDH-L.
- Full first-layer neuron activations may be captured through an optional
  analysis hook. They should not be retained during normal training.

## Projection design

- `P_value` is an independent `Linear(channels, d_model)` projection.
- `P_sparse` maps `channels` to `n_neurons`.
- For small models, `P_sparse` may be dense.
- For large models, use an efficient factorized projection:

```text
P_sparse(m) = A(B(m))
B: channels -> projection_rank
A: projection_rank -> n_neurons
```

- Do not use slow, unstructured sparse GPU operations by default.
- Do not derive the value as `E(s_t)` in the primary model. Tying the input to
  BDH's shared output matrix tests an additional idea and belongs in a later
  `BDH-S-tied` ablation.

## Model requirements

- Only the first BDH layer bypasses its normal `decoder_x`; it uses `s_t`
  directly as its sparse query and key.
- The first layer's memory update pairs the rotated `s_t` with `v_t`.
- Its output gate, residual path, masking, reset behavior, and state shape stay
  consistent with the existing BDH implementation.
- Later layers remain standard `ContinuousBDH` layers.
- The value and sparse projections train jointly with the BDH core.
- Do not add a CNN, task head, or source-localization system to this model.

## Interpretability outputs

BDH-S must make it possible to measure:

- each first-layer neuron's activation over time;
- the MEG sensor weights that contribute to that neuron;
- event or phoneme selectivity; and
- prediction changes when selected neurons are disabled.

Neuron indices do not have a natural 2D layout. A 2D page must arrange neurons
using measured activation similarity, learned sensor-weight similarity, or a
separately specified topographic training rule. Simply reshaping the neuron
vector into a square is not a meaningful brain map.

## Acceptance checks

BDH-S is ready when:

1. Both input paths have the documented shapes and receive gradients.
2. The first layer uses the supplied sparse activation and does not call its
   normal `decoder_x` for that layer.
3. Later layers follow the unchanged standard BDH calculation.
4. Full-sequence and chunked execution agree when dropout is disabled.
5. Masks and per-stream resets behave exactly as in BDH-L.
6. The optional analysis hook reproduces the first-layer activation used by
   the memory update without changing predictions.
7. Dense and factorized projections pass parameter-count and shape tests.
8. A real LibriBrain batch completes forward, loss, backward, optimizer step,
   checkpoint, and reload.

## Primary comparison

Compare BDH-S with BDH-L using the same data, preprocessing, task, context,
training budget, and total parameter budget. Treat dense, factorized,
sensor-masked, tied, and 2D-topographic forms as separate ablations rather
than combining them into one model.
