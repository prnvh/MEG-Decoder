# MEG-BDH Project Build Specification — Parallel Execution Model

## Core Execution Principle

The project should **not** operate as:

```text
Preprocessing
    ↓
Engineering
    ↓
ML
    ↓
Evaluation
```

Instead, it should operate as several parallel workstreams with defined synchronization points:

```text
                    PROJECT START
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
 ENGINEERING       PREPROCESSING       RESEARCH / YOU
       │                 │                 │
       │                 │                 │
       ├──────┐     ┌────┤                 │
       │      │     │    │                 │
       ▼      ▼     ▼    ▼                 ▼
  BDH CORE  MOCK   DATA  EVAL          EXPERIMENT
  WORK      DATA   PIPE  DESIGN          DESIGN
       │      │     │    │                 │
       └──────┼─────┼────┼─────────────────┘
              │     │    │
              ▼     ▼    ▼
            INTEGRATION v0
                   │
                   ▼
              ML SMOKE TESTS
                   │
          ┌────────┼────────┐
          │        │        │
          ▼        ▼        ▼
       baseline   BDH     pipeline
       training   tests    debugging
          │        │        │
          └────────┼────────┘
                   ▼
              DATA FREEZE v1
                   +
              HARNESS v1
                   │
                   ▼
          FORMAL EXPERIMENTS
             RUN IN PARALLEL
```

The rule is:

> **Only block work when it genuinely depends on something unfinished. Otherwise run it in parallel.**

---

# Parallel Workstreams

## Workstream A — Engineering

Engineering starts immediately.

Engineering does **not** need to wait for LibriBrain preprocessing to finish.

### A1. BDH investigation

Immediately:

* reproduce original BDH
* map architecture
* identify text-specific components
* identify persistent-state components
* identify large parameter blocks
* document tensor shapes
* expose configuration

This can happen while the dataset is downloading and being audited.

---

### A2. Continuous-input BDH

Engineering builds:

```text
[B, T, D]
    │
    ▼
BDH core
    │
    ▼
[B, T, D_model]
```

using **synthetic tensors initially**.

There is no need to wait for real MEG.

Example development input:

```python
x = torch.randn(
    batch_size,
    time,
    latent_dim
)
```

This lets Engineering test:

* forward pass
* backward pass
* persistent state
* memory consumption
* different BDH sizes
* checkpointing
* state reset
* state continuation

before LibriBrain is ready.

---

### A3. Training harness

Also begins immediately.

Use synthetic datasets initially.

Engineering builds:

```text
configuration
     ↓
training
     ↓
validation
     ↓
checkpoint
     ↓
evaluation
```

The harness should be largely model-agnostic.

---

### A4. MEG encoder interface

Engineering and Preprocessing agree early on an interface:

```text
INPUT

[B, T, 306]
```

or another clearly defined arrangement.

Engineering can build against this contract using fake data.

Later real LibriBrain batches simply replace fake batches.

---

# Workstream B — Preprocessing

Preprocessing also starts immediately.

## B1. LibriBrain acquisition

Parallel with Engineering:

```text
download
dataset documentation
licensing
file inspection
metadata audit
```

---

## B2. Minimal dataset first

Do **not** wait until the complete preprocessing system is perfect.

Produce an early:

```text
DATASET v0
```

containing perhaps:

```text
one session
small number of events
MEG tensor
word timestamps
phoneme timestamps
```

Its purpose is **integration**, not research.

Engineering and ML use this to test the real pipeline.

---

## B3. Full preprocessing continues

While v0 is already being used:

```text
normalization
resampling
session indexing
word alignment
phoneme alignment
continuous windows
semantic targets
integrity checks
```

continue in parallel.

Eventually this becomes:

```text
DATASET v1
```

---

# Workstream C — ML

ML should not be idle while preprocessing is happening.

The distinction is:

> **ML development starts early. Formal experiments start after the dataset is stable.**

---

## C1. Before real data

ML can already work on:

* loss design
* optimization strategy
* baseline architecture
* semantic objective
* masked prediction objective
* training configs
* experiment matrix
* expected metrics
* debugging procedures

---

## C2. Once Dataset v0 exists

ML immediately begins smoke testing.

For example:

```text
tiny LibriBrain sample
        │
        ▼
MEG encoder
        │
        ▼
simple classifier
```

Question:

> Can we overfit 100 examples?

This is an extremely useful early test.

If the model cannot deliberately overfit a tiny dataset, something is probably broken.

---

## C3. Baseline development

The baseline does not need the full BDH architecture.

As soon as usable data exists:

```text
MEG
 ↓
simple encoder
 ↓
phoneme classifier
```

can be trained.

This lets preprocessing and ML debug each other.

---

## C4. BDH smoke tests

As soon as Engineering provides continuous-input BDH:

```text
Dataset v0
    +
MEG encoder
    +
BDH
```

can be integrated.

Again, these results are **not final research results**.

Their job is to find problems early.

---

# Workstream D — You

Your work runs across everything continuously.

You should not wait for implementations to finish.

Initially you should be working on:

```text
research hypothesis

BDH architecture understanding

MEG-BDH interface

baseline selection

evaluation methodology

ablation design

interpretability methodology

patent-relevant architecture questions
```

while the others are building.

---

# Workstream E — Evaluation

Evaluation should also begin early.

Do not build evaluation after training finishes.

Preprocessing + Engineering + You should define:

```text
metrics

held-out sessions

evaluation formats

prediction files

baseline comparisons

context curves

compute metrics
```

before serious model training.

This prevents:

> We trained a model; now how do we evaluate it?

---

# Dependency Model

Rather than one dependency chain, use **hard dependencies and soft dependencies**.

## Hard dependency

Work genuinely cannot proceed.

Example:

```text
Formal word decoding
        │
        requires
        ▼
validated word alignment
```

ML must wait.

---

## Soft dependency

Real data is not ready, but development can proceed.

Example:

```text
BDH training harness
```

does not require final LibriBrain.

Use:

```text
synthetic MEG
```

or:

```text
Dataset v0
```

and continue.

---

# Project Synchronization Points

Use a small number of integration gates rather than sequential phases.

---

## Sync Point 0 — Interface Freeze

Very early.

Agree on:

```text
MEG tensor shape

target representation

mask representation

session ID

timestamps

metadata

model input/output
```

For example:

```python
batch = {
    "meg": Tensor[B, T, 306],
    "mask": Tensor[B, T],
    "target": ...,
    "session_id": ...,
    "metadata": ...
}
```

Once this is agreed, teams can work independently.

---

# Parallel Block 1 — Foundation

All four roles work simultaneously.

```text
ENGINEERING
│
├── reproduce BDH
├── isolate core
├── synthetic input
└── training harness


PREPROCESSING
│
├── acquire LibriBrain
├── inspect dataset
├── understand timestamps
└── create Dataset v0


ML
│
├── define losses
├── simple baseline
├── training configs
└── experiment design


YOU
│
├── architecture
├── evaluation design
├── research questions
└── BDH modification strategy
```

---

# Sync Point 1 — First Integration

Required:

```text
Dataset v0
+
MEG encoder v0
+
BDH core v0
+
trainer v0
```

Run:

```text
real LibriBrain batch
        ↓
MEG encoder
        ↓
BDH
        ↓
loss
        ↓
backward
```

Success criterion:

> One complete training step on real LibriBrain.

This should happen **well before preprocessing is finished**.

---

# Parallel Block 2 — Pipeline Development

After first integration:

```text
PREPROCESSING
│
├── build full session processing
├── normalization
├── word alignment
├── phoneme alignment
├── caching
└── validation


ENGINEERING
│
├── optimize BDH
├── configuration
├── checkpointing
├── profiling
├── BDH size variants
└── evaluation harness


ML
│
├── overfit tiny dataset
├── baseline experiments
├── BDH smoke tests
├── optimizer testing
└── objective testing


YOU
│
├── inspect failures
├── review state behaviour
├── refine architecture
└── lock formal experiments
```

Everything happens together.

---

# Sync Point 2 — Data Freeze v1

At this point Preprocessing certifies:

```text
session splits

normalization

alignment

context extraction

labels

evaluation sets
```

This means results after this point can be treated as research results.

---

# Important Distinction

**Data Freeze v1 does not mark the start of all ML.**

It marks the start of:

# FORMAL ML EXPERIMENTS

ML development and debugging should already have been happening.

---

# Parallel Block 3 — Formal Experiments

Once Data v1 and Harness v1 are stable, experiments should themselves run in parallel where compute allows.

For example:

```text
GPU 1
Encoder baseline

GPU 2
BDH-small 2s

GPU 3
BDH-small 30s

GPU 4
Transformer 30s
```

rather than:

```text
baseline
   ↓
wait
   ↓
BDH 2s
   ↓
wait
   ↓
BDH 30s
```

Independent experiments should be parallelized.

---

# Formal Experiment Wave 1

Run simultaneously:

```text
E001
Encoder baseline — phoneme — 2s

E002
BDH-small — phoneme — 2s

E003
Encoder baseline — word — 2s

E004
BDH-small — word — 2s
```

Purpose:

> Determine whether BDH works at all.

---

# Formal Experiment Wave 2

If Wave 1 works:

```text
BDH 0.5s

BDH 10s

BDH 30s

BDH 60s

BDH 150s
```

can largely run simultaneously.

In parallel:

```text
Transformer 2s

Transformer 30s

Transformer 150s
```

---

# Formal Experiment Wave 3

Once a useful BDH configuration emerges, parallelize architecture ablations:

```text
BDH tiny

BDH small

BDH medium

BDH full
```

At the same time, another workstream can start:

```text
semantic alignment
```

while you begin analyzing:

```text
internal state
```

from already completed checkpoints.

---

# Evaluation Runs Continuously

Evaluation should not wait until the end.

Every finished checkpoint automatically goes:

```text
checkpoint
    │
    ├── validation metrics
    ├── compute profile
    ├── context metrics
    └── experiment registry
```

Then later:

```text
strong checkpoints
       │
       ▼
deeper evaluation
       │
       ├── held-out session
       ├── semantic retrieval
       └── interpretability
```

---

# Interpretability Can Also Start Early

You do not need the final best model.

Once the first useful BDH model exists:

```text
BDH-small checkpoint
        │
        ▼
state extraction
```

can begin.

This lets you discover early whether:

```text
state persistence

sparsity

concept selectivity
```

are behaving as expected.

Those results can influence Engineering while architecture work is still ongoing.

This feedback loop is important:

```text
ENGINEERING
    │
    ▼
MODEL
    │
    ▼
INTERPRETABILITY
    │
    ▼
OBSERVATION
    │
    ▼
ENGINEERING CHANGE
```

---

# BDH Compression Should Run Alongside ML

Do not wait until all decoding experiments are complete to remove BDH capacity.

Once BDH-small trains:

```text
Engineering + You
        │
        ▼
inspect architecture
        │
        ▼
build BDH-tiny / medium
```

while ML continues training existing models.

For example:

```text
ML:
BDH-small context experiments

          simultaneously

Engineering:
BDH-tiny implementation

          simultaneously

Preprocessing:
semantic dataset improvements

          simultaneously

You:
state analysis + evaluation
```

---

# Revised High-Level Schedule

The actual structure should look closer to:

```text
TIME ───────────────────────────────────────────────►


ENGINEERING
BDH analysis ── continuous input ── harness ── compression ── optimize
████████████████████████████████████████████████████████████████


PREPROCESSING
download ─ audit ─ dataset v0 ─ full pipeline ─ eval support
█████████████████████████████████████████████████


ML
design ─ smoke ─ baseline ─ BDH test ─ formal runs ─ semantic
     ███████████████████████████████████████████████████████


YOU
architecture ─ integration ─ evaluation ─ ablation ─ interpretation
████████████████████████████████████████████████████████████████


EVALUATION
design ─ sanity metrics ─ validation ─ comparison ─ deep analysis
   ██████████████████████████████████████████████████████████
```

That is much closer to how this should actually run.

---

# What Is Truly Sequential

Only a few things need strict ordering.

### Formal training requires validated data

```text
Data Freeze v1
      ↓
formal experiments
```

### Semantic training requires semantic targets

```text
semantic target construction
        ↓
semantic experiment
```

### Final test evaluation requires model selection

```text
validation
    ↓
model selection
    ↓
held-out test
```

### Interpretation claims require trained models

```text
checkpoint
    ↓
state extraction
    ↓
interpretability
```

Everything else should be parallelized wherever practical.

---

# Role Interaction Model

Instead of handoffs:

```text
A → B → C
```

use continuous feedback:

```text
             ENGINEERING
             ↗         ↘
            ↕           ↕
          YOU ←───────→ ML
            ↕           ↕
             ↖         ↗
            PREPROCESSING
```

Examples:

### Preprocessing ↔ ML

ML discovers a certain normalization behaves badly.

Preprocessing investigates and updates it.

### ML ↔ Engineering

ML observes unstable BDH gradients.

Engineering investigates state mechanics.

### You ↔ Engineering

Interpretability suggests half the latent population is unused.

Engineering creates a smaller BDH.

### Evaluation ↔ ML

Long-context gain disappears in held-out sessions.

ML changes the experiment rather than blindly continuing.

---

# Revised Responsibility Philosophy

### Engineering

Stay approximately **one step ahead of ML**.

ML should rarely be waiting for infrastructure.

### Preprocessing

Stay approximately **one dataset version ahead of ML**.

Give ML a usable imperfect version early, then stabilize it.

### ML

Begin experimentation early but distinguish:

```text
development results
```

from:

```text
formal results
```

### You

Continuously connect all workstreams and redirect effort based on evidence.

---

# Critical Path

The true critical path is much smaller than the whole project:

```text
LibriBrain access
       ↓
basic preprocessing
       ↓
Dataset v0
       ↓
first real batch
       ↓
MEG encoder + BDH
       ↓
successful backward pass
       ↓
validated Dataset v1
       ↓
formal BDH training
       ↓
held-out evaluation
```

Everything not on that path should be pushed into parallel work.

---

# Practical Rule

At every team check-in, ask:

> **Is anyone waiting for another person when they could be building against an interface, mock dataset, previous dataset version or existing checkpoint instead?**

If yes, restructure the work.

The aim should be to keep all four roles moving simultaneously while using a few carefully chosen synchronization points to preserve experimental validity.
