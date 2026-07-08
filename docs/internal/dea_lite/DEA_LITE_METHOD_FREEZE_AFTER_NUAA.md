# DEA-lite Method-Level Freeze After NUAA

## Decision

DEA-lite is frozen as pilot / ablation / limitation evidence.

It is not the full DEA method.

## Evidence status

| Dataset | Status | Interpretation |
|---|---:|---|
| NUDT-SIRST | Positive | DEA-lite 0.005 improves the MSHNet baseline under the recorded paired setting. |
| IRSTD-1K | Positive signal | DEA-lite 0.005 shows a false-alarm-control / PD-FA trade-off signal. |
| NUAA-SIRST | Stable negative | DEA-lite 0.005 fails the paired gate and exposes dataset-dependent behavior. |

## NUAA negative result

```text
gate_pass = false
decision = DEA_LITE_NEGATIVE_DATASET_DEPENDENT

Delta:
  IoU -0.0336
  PD  -0.0266
  FA  +2.2104

epoch audit:
  num_gate_pass_epochs = 0
```

## Method interpretation

The current implementation should be described as:

```text
DEA-lite = MSHNet + evidence-aware training regularization / loss constraints.
```

It should not be described as:

```text
full DEA architecture
explicit evidence decomposition model
counterfactual intervention method
inference-time evidence-control method
```

## Why this matters

The NUAA negative result shows that loss-level evidence regularization alone is insufficient.

This motivates a future Full DEA design with:

```text
target evidence branch
clutter evidence branch
counterfactual suppression or swapping path
dual real/counterfactual prediction
inference-time evidence gate or evidence-calibrated segmentation head
```

## Forbidden claims

Do not claim:

```text
DEA-lite validates full DEA.
DEA-lite is the proposed full DEA method.
DEA-lite universally improves IRSTD.
DEA-lite solves false alarms.
DEA-lite is AAAI-ready as a full method.
DEA-lite 0.005 is globally robust.
```

## Allowed claims

Allowed:

```text
DEA-lite is a lightweight evidence-aware training regularizer on top of MSHNet.
DEA-lite shows promising behavior on NUDT-SIRST and IRSTD-1K.
DEA-lite fails on NUAA-SIRST, revealing dataset-dependent limitations.
DEA-lite motivates the need for a full DEA architecture with explicit evidence decomposition and counterfactual control.
```

## Current route

```text
1. Freeze DEA-lite evidence and claims.
2. Stop presenting DEA-lite as full DEA.
3. Start a separate Full DEA predeclare protocol.
4. Implement Full DEA only after protocol review.
```
