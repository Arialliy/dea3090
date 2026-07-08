# Full DEA Predeclare Protocol

> This document is a protocol declaration only.
> It does not implement Full DEA code.

## Motivation

DEA-lite is frozen as pilot / ablation / limitation evidence.

DEA-lite shows:

```text
NUDT-SIRST: positive anchor
IRSTD-1K: false-alarm-control signal
NUAA-SIRST: stable negative result
```

The NUAA failure suggests that loss-level evidence regularization is insufficient.

## Full DEA Hypothesis

A full DEA method should explicitly model and intervene on evidence, rather than only regularizing training loss.

## Required Structural Components

A Full DEA implementation must include:

```text
1. target evidence branch
2. clutter evidence branch
3. counterfactual intervention path
4. real prediction and counterfactual prediction
5. inference-time evidence gate or evidence-calibrated segmentation head
```

## Non-Goals

Full DEA must not be implemented as merely:

```text
MSHNet + another scalar loss
MSHNet + lambda tuning
DEA-lite with a new lambda
post-hoc threshold adjustment
dataset-specific lambda selection
```

## First Gate Dataset

Use NUAA-SIRST first.

Reason:

```text
NUAA is the dataset where DEA-lite 0.005 failed stably.
A full DEA design should first show that explicit evidence decomposition and counterfactual control can fix this failure mode.
```

## Baselines

Compare:

```text
MSHNet baseline
DEA-lite 0.005 negative result
Full DEA candidate
```

## First Gate

Full DEA on NUAA-SIRST must satisfy:

```text
IoU >= MSHNet baseline
PD  >= MSHNet baseline - tolerance
FA  <= MSHNet baseline
and Full DEA must outperform DEA-lite 0.005 on NUAA.
```

## Evidence Rules

Do not claim Full DEA works until:

```text
1. Full DEA code is predeclared and committed.
2. NUAA seed result passes the first gate.
3. The same protocol is reproduced on NUDT-SIRST and IRSTD-1K.
4. Failure analysis confirms reduced false alarms without target collapse.
```

## AAAI Route

Full DEA may become an AAAI route only after it has:

```text
explicit architecture contribution
counterfactual/evidence-control mechanism
NUAA recovery evidence
multi-dataset paired evidence
ablation separating target evidence, clutter evidence, and counterfactual control
```

## Implementation Boundary

This branch must not add:

```text
FullDEAHead
full_dea_loss
counterfactual branch code
inference-time evidence gate code
training scripts for Full DEA
```

The next implementation branch may only start after this protocol is reviewed.
