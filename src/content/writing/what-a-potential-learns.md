---
title: What a machine-learned potential actually learns
date: 2026-07-14
kicker: Descriptors, smoothness, and why transferability is mostly a data problem.
tags: [interatomic-potentials, machine-learning]
math: true
dateless: true
stub: true
stubNote: An outline rather than a finished piece.
---

## The descriptor is the whole game

A potential can only distinguish two environments that its descriptor distinguishes.
Everything downstream is constrained by that one choice: the regression, the training
set, the validation protocol. It is also the part most papers spend the least time on.

## Smoothness is not a nice-to-have

Outline: why discontinuities in the descriptor become discontinuities in the force,
and what that does to a molecular dynamics trajectory that has to conserve energy for
a few million steps.

## Transferability is mostly a data problem

Outline: the argument that models fail outside their training distribution in ways
that look like model failures and are actually sampling failures. Needs a worked
example. The amorphous carbon case is the obvious one, and the hero figure on the
landing page is already the right illustration.

## What to write next

- A concrete descriptor comparison, with a figure generated from a committed script.
- The honest version of "how do you know when you are extrapolating".
- Set `math: true` is already on; add the actual equations.
