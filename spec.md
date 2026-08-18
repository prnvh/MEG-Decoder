Project: CNN + BDH for Long-Context MEG Decoding

Core question:
Can BDH's persistent state improve language decoding from MEG when compared with conventional temporal models, particularly as the amount of neural context increases?

Use LibriBrain as the main dataset. A small CNN extracts local MEG features; BDH is responsible for modelling the long temporal sequence.

LibriBrain MEG
306 channels
     │
     ▼
small CNN
local neural features
     │
     ▼
BDH
long-context + persistent state
     │
     ▼
decoder head
     │
     ├── phoneme
     └── word

The paper is not about thought decoding. It is about whether BDH is a useful persistent-state architecture for long-context neural decoding.

1. Preprocessing

Keep preprocessing standardized rather than making it part of the contribution.

Start from LibriBrain's released MEG.

Pipeline:

LibriBrain
    │
    ▼
session-separated train / val / test
    │
    ▼
channel normalization
statistics from TRAIN only
    │
    ▼
resampling
    │
    ▼
model-ready MEG
[B, 306, T]
    │
    ▼
align with phoneme / word timestamps
Main preprocessing choices

Use:

existing LibriBrain session-level splits
no random adjacent-window splitting
normalization using training sessions only
50 Hz as the main long-context representation
optionally 100 Hz as a sensitivity check
preserve continuous temporal ordering
construct windows based on target events

For a word occurring at time t:

previous context                    target
─────────────────────────────────────│──────
                                     t

The model receives the preceding MEG plus activity around the event.

Context lengths:

2 s
10 s
30 s
60 s
150 s

You could include 0.5 s later, but those five are enough for the main scaling experiment.

1. CNN

The CNN is intentionally not the main model.

Its only job is local MEG feature extraction.

Something like:

[B,306,T]
     │
     ▼
Conv1D
306 → 256
     │
     ▼
activation
     │
     ▼
1–2 residual temporal conv blocks
     │
     ▼
[B,T,256]

Keep its temporal receptive field relatively small.

The division should be:

# CNN

milliseconds / local signal structure

# BDH

seconds → minutes / sequence state

Avoid a large CNN because then it becomes unclear whether BDH contributes anything.

1. BDH

BDH is the main experimental model.a

Input:

z1 z2 z3 ... zT

from the CNN.

BDH maintains its state across that sequence:

z1 ──►
z2 ──► BDH persistent state
z3 ──►
...
zT ──►

For Paper 1, don't radically redesign BDH.

Remove whatever text-specific interface is necessary, but preserve the main:

sparse activation mechanism
temporal/state computation
persistent state
recurrent/state behaviour

That makes the paper's question clean:

Does the existing BDH state mechanism transfer usefully to continuous MEG?

1. Decoder heads

Keep these tiny.

Phoneme head
BDH state
    ↓
linear / small MLP
    ↓
phoneme class

This is primarily a sanity benchmark.

Word head
BDH state
    ↓
projection
    ↓
word class / word embedding

Word decoding should be the main task.

Phonemes show that the system works at the lower linguistic level; words provide the more interesting long-context test.

I would leave full semantic decoding for either an additional experiment or Paper 2 unless it comes cheaply.

1. Training

For the first paper, keep training straightforward.

I would not make self-supervised pretraining necessary for the first result.

Start end-to-end:

MEG
 ↓
CNN
 ↓
BDH
 ↓
head
 ↓
loss
 ↓
backprop through everything

Train:

CNN + BDH + head

together.

Use normal controls:

AdamW
mixed precision
gradient clipping
validation-based checkpoint selection
early stopping
fixed random seeds
several seeds for major experiments

The main thing ML needs to vary is:

architecture
context length
memory condition

not dozens of hyperparameters.

1. Main Experiment — Context Length

This is the central result.

Train/evaluate the same model with:

2 s
10 s
30 s
60 s
150 s

Then plot:

decoding performance
      ▲
      │
      │                    BDH
      │              ●─────●
      │         ●────
      │    ●────
      │ ●
      └────────────────────────►
          neural context

Your main hypothesis is:

BDH should benefit disproportionately from longer context.

It does not necessarily need to dominate at 2 seconds.

It becomes interesting if the gap increases at 30–150 seconds.

1. Main Ablation — BDH Memory

This is arguably as important as the baseline comparison.

For a long sequence, compare:

BDH state maintained continuously

vs

state reset every 30 s

vs

state reset every 10 s

vs

state reset every 2 s

Everything else stays identical.

If performance falls as memory is shortened:

persistent      best
30 s reset       ↓
10 s reset       ↓
2 s reset        ↓

then you have direct evidence that persistent state is actually contributing.

Without this experiment, reviewers can reasonably ask whether BDH is merely another nonlinear network sitting behind the CNN.

1. Baselines

The crucial rule:

Use the same CNN front end for every sequence-model baseline.

So:

```
              ┌──► head
              │
```

MEG → same CNN ───┼──► GRU/LSTM ─────► head
                  │
                  ├──► Transformer ───► head
                  │
                  └──► BDH ───────────► head
Baseline A — CNN only

Tests:

Does temporal modelling beyond the CNN actually matter?

Very important.

Baseline B — CNN + GRU or LSTM

Tests BDH against traditional persistent recurrent memory.

You only need one of GRU/LSTM. I'd probably use GRU to keep it simpler.

Baseline C — CNN + Transformer

This should be the principal modern baseline.

Try to approximately parameter-match it to BDH.

For example:

CNN + Transformer    ~20–30M
CNN + BDH            ~20–30M

It doesn't have to be mathematically exact, but don't compare a 3M Transformer against a 30M BDH.

1. Model Roles

So the paper contains four actual systems:

Model	Purpose
CNN	Local-feature baseline
CNN + GRU	Conventional recurrent-memory baseline
CNN + Transformer	Modern sequence baseline
CNN + BDH	Proposed model

BDH receives the most analysis because it's the model under study.

The CNN stays identical wherever possible.

1. Evaluation

Your primary evaluation should be on held-out LibriBrain sessions.

For phonemes:

balanced accuracy
F1

For words:

top-1
top-5
top-10
potentially retrieval rank

Also report:

parameter count
GPU memory
training throughput
inference cost

because BDH could be interesting even if raw decoding performance is similar but its long-context behaviour is better.

1. Minimum Experiment Matrix

You don't need hundreds of runs.

A reasonable first paper could center on:

Model	2s	10s	30s	60s	150s
CNN	✓	✓	✓	✓	✓
CNN + GRU	✓	✓	✓	✓	✓
CNN + Transformer	✓	✓	✓	✓	✓
CNN + BDH	✓	✓	✓	✓	✓

Then for BDH:

Memory	Evaluation
Persistent	✓
Reset every 30s	✓
Reset every 10s	✓
Reset every 2s	✓

Run the main table for word decoding and repeat the most important comparisons for phonemes.

That is already a substantial experimental package.

1. What Makes the Paper Publishable

The paper should ideally establish three things:

Result 1 — BDH works on neural sequences
MEG → CNN → BDH

successfully learns LibriBrain decoding.

Result 2 — Long context matters

Performance changes systematically as:

2 → 10 → 30 → 60 → 150 seconds
Result 3 — BDH state matters

Resetting BDH's persistent state reduces the advantage.

If you additionally get:

Result 4 — BDH beats/matches Transformer more strongly at long context

then you have a particularly clean paper.