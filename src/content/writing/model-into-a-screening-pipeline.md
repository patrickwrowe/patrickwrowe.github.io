---
title: Getting a model into a screening pipeline
date: 2026-05-19
kicker: The gap between a good validation number and a decision someone will act on.
tags: [ml-engineering, drug-discovery]
stub: true
stubNote: An outline rather than a finished piece.
---

## The validation number is the easy part

A model that ranks well on a held-out split has cleared the lowest bar in the
process. The question that decides whether it gets used is different: when this
model and an experienced scientist disagree, who is right, and how would anyone know?

## Calibration beats accuracy

Outline: why a well-calibrated mediocre model is more useful in a screening cascade
than a sharp uncalibrated one, because the downstream decision is a threshold and a
threshold needs a probability that means something.

## The split is the experiment

Outline: random splits flatter models. Domain-appropriate splits, whether scaffold,
temporal or target-level, are what tell you whether the thing will work on next
month's chemistry.

## Trust is built at the boundary

Outline: what it takes for a screening team to act on a prediction. Abstention,
uncertainty that is legible, and being visibly right about the cases the team already
knows the answer to.

## What to write next

- A concrete cascade with real numbers, if any can be cleared for external use.
- The failure case: a model that was accurate, well-engineered, and never used.
