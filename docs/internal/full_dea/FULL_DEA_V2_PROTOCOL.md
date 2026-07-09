# Full DEA v2 Protocol

This protocol supersedes the prototype-only Full DEA head for the next
implementation step. It does not report any positive result.

## Method Boundary

Full DEA v2 is:

```text
MSHNet multi-scale fusion
+ baseline-preserving target residual
+ explicit target / clutter evidence decomposition
+ hard false-alarm pseudo clutter supervision
+ subtractive counterfactual clutter suppression
```

Full DEA v2 is not:

```text
DEA-lite lambda tuning
post-hoc threshold selection
an x_d0-only attention head
a claim of success before NUAA-first gate passes
```

## Fixed First Gate

Use NUAA-SIRST first because DEA-lite failed there.

Reference gate:

```text
MSHNet baseline:
  IoU 0.7461767423
  PD  0.9619771863
  FA  25.3124771831

Full DEA v2 must satisfy:
  IoU >= 0.7461767423
  PD  >= 0.9569771863
  FA  <= 25.3124771831
```

Single-seed success on NUAA is only a first-gate pass. It is not a broad
method claim.

## P0 Validation Before Long Training

Before any 400/500 epoch training, run:

```text
python compile checks
FullDEAHeadV2 shape and baseline-initialization tests
native multi-scale mask contract tests
hard clutter ratio bound tests
main argument guard tests
one-epoch smoke only
```

## Training Control

Preferred initialization:

```text
load MSHNet NUAA best checkpoint with strict=False
missing keys must be limited to full_dea_head.*
```

Preferred warm behavior:

```text
Full DEA v2 should not waste early epochs in output_0-only warm-up.
Use --full-dea-start-epoch and --full-dea-ramp-epochs for auxiliary loss ramp.
```

Preferred fairness note:

```text
If Full DEA v2 uses MSHNet pretraining plus finetuning, paper comparisons need
a matched MSHNet finetune/control budget before main-table claims.
```
