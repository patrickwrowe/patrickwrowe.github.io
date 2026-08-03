---
title: SMILES is a strange language to model
date: 2026-06-09
kicker: Canonicalisation, invalid strings, and what tokenisation costs you.
tags: [generative-models, cheminformatics]
stub: true
stubNote: >-
  Title and kicker are from the prototype; the prose below is an outline, not a finished piece.
---

## It looks like text and it is not

SMILES is a serialisation of a graph. Two strings that differ everywhere can denote
the same molecule, and two strings that differ by one character can denote molecules
that behave nothing alike. A language model trained on it inherits both problems.

## Canonicalisation is a modelling decision

Outline: training on canonical SMILES only versus augmenting with randomised
traversals; what each does to the effective size of the training set and to what the
model learns about the underlying graph.

## Invalid strings are information

Outline: validity rate as a metric is less useful than it looks. A model at 95%
validity and a model at 99% can be differently wrong, and the interesting question is
what the invalid 1% has in common.

## What tokenisation costs

Outline: character-level versus atom-level versus learned subwords, and the
ring-closure digits that break all three.

## What to write next

- Measured numbers from the model behind the molecular generation project, once
  those exist and can be quoted honestly.
- A figure showing the failure modes, generated from a committed script.
