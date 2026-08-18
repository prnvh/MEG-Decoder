# BDH-L specification

Status: proposed

## Purpose

BDH-L is the simple MEG version of BDH. It maps each MEG sample into BDH's
small working space, then runs the existing BDH model without changing its
memory mechanism.

The `L` means **latent entry**: MEG enters through the model's compact latent
representation rather than directly activating its sparse neurons.

## Data flow

```text
MEG sample [channels]
        -> learned linear projection
latent value [d_model]
        -> standard ContinuousBDH
hidden output [d_model] + recurrent state
```

For MEG sample `m_t`:

```text
v_t = P_meg(m_t)
output_t, state_t = BDH(v_t, state_(t-1))
```

The existing `ContinuousBDH` input normalization supplies the required
normalization. The boundary projection must not add a second LayerNorm.

## Interface

- Input: MEG tensor `[batch, time, channels]`.
- Expected channel count: read from the dataset manifest; do not hard-code 306.
- Optional inputs: valid-sample mask, incoming state, stream reset mask, and
  state detachment flag.
- Output: hidden tensor `[batch, time, d_model]`, next BDH state, and existing
  BDH diagnostics.
- A task head, loss function, label handling, and temporal downsampling remain
  outside this model.

## Model requirements

- Add one learned `Linear(channels, d_model)` projection at the input boundary.
- Keep all standard BDH layers, shared weights, causal reads, state updates,
  masking, resets, and chunked execution unchanged.
- Do not add a CNN, temporal encoder, or task-specific prediction head.
- The projection and BDH core must train together.
- Running inference must not permanently update model weights. Permanent
  person-specific learning happens only during training or fine-tuning.

## Configuration

Required settings:

- `input_channels`
- `d_model`
- `n_neurons`
- `n_heads`
- `n_layers`
- existing dropout, state-decay, and RoPE settings

The additional parameter count is `input_channels * d_model + d_model` when
the projection uses a bias.

## Acceptance checks

BDH-L is ready when:

1. It accepts `[B, T, channels]` and returns `[B, T, d_model]`.
2. Gradients reach both the MEG projection and every BDH weight.
3. Processing a sequence in chunks matches processing it in one pass when
   dropout is disabled.
4. Invalid samples do not read or change recurrent state.
5. Resetting one stream does not reset other streams in the batch.
6. Existing `ContinuousBDH` tests continue to pass.
7. A real LibriBrain batch completes forward, loss, backward, optimizer step,
   checkpoint, and reload.

## Primary comparison

Compare BDH-L against a parameter-matched linear-input GRU first. A CNN-GRU is
a stronger practical baseline, but it also has a stronger input processor and
therefore does not isolate the memory mechanism.
