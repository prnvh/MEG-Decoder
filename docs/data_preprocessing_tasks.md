# Data and preprocessing tasks

Source: trimmed from [`BUILD.md`](../BUILD.md). The authoritative dataset,
timing, split, and causality rules are in
[`RESEARCH_CONTRACT.md`](../RESEARCH_CONTRACT.md).

## Outcome

Provide one reproducible stream of cleaned, aligned MEG data that all newly
trained models consume without architecture-specific changes.

## Acquisition and audit

- Acquire LibriBrain under its declared licence and record source versions.
- Inspect every session's channel count, channel order, sensor types, sampling
  rate, duration, timestamps, event files, and missing data.
- Confirm what the released `bads+headpos+sss+notch+bp+ds` preprocessing has
  already performed.
- Produce a manifest instead of hard-coding assumptions such as 306 channels or
  250 Hz.
- Report unusual sessions, channels, artifacts, or event inconsistencies for
  review rather than silently dropping them.

## Dataset v0: integration data

Release a small, non-final dataset containing:

- one or more sessions;
- chronological MEG tensors;
- phoneme events and timestamps;
- masks and session boundaries; and
- enough examples to test loading, forward, loss, backward, and overfitting.

Dataset v0 exists for integration. Results produced from it are not formal
research results.

## Dataset v1: frozen research data

- Split by whole recording or session before normalization and window creation.
- Fit normalization statistics using training data only.
- Maintain an exact `cnn_compat` view for evaluating the external PNPL CNN.
- Maintain a shared `continuous_causal` view for newly trained CNN, recurrent,
  BDH-L, and BDH-S models.
- Align phoneme events with a declared neural-response delay selected without
  test data.
- Apply anti-alias filtering before any resampling; never decimate by simply
  skipping samples.
- Build chronological 0.5-, 2-, and 30-second inputs as required by the locked
  experiment definitions.
- Reset streams at recording/session boundaries and mark padding with masks.
- Cache processed data without losing original timestamps or event IDs.
- Hash the source manifest, preprocessing configuration, splits, and processed
  outputs.

## Batch contract

Each batch must provide at least:

```text
meg                 [batch, time, channels]
sample_mask         [batch, time]
target              task-defined labels
target_mask         valid prediction positions
time_seconds        original sample times
event_id            stable event identifiers
subject_id
session_id
recording_id
reset_mask          new-stream indicators
```

Any alternate orientation required by an external checkpoint is created by its
adapter, not stored as a second source of truth.

## Quality report

Generate automated summaries for:

- channel variance and missing values;
- representative sensor traces and power spectra;
- artifact and rejection counts;
- label counts and duration distributions;
- event-to-sample alignment;
- split durations and class balance; and
- differences between train, validation, and test distributions.

Human review approves rules and flagged cases; the pipeline applies the frozen
rules consistently to all recordings.

## Handoffs

Data provides:

1. Dataset v0 for early integration;
2. a manifest-backed batch that satisfies the interface;
3. the audit and quality report;
4. Dataset v1 with frozen splits, timing, normalization, and hashes; and
5. deterministic commands to rebuild every processed artifact.

Formal training cannot begin until Dataset v1 passes integrity checks and its
manifest, split, response delay, and preprocessing configuration are frozen.
