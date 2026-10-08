# Transformer Attention Demo

We are building an interactive visualization for a graduate Deep Learning
presentation on Vaswani et al. (2017), "Attention Is All You Need."

## Source files

Read these before making architectural or mathematical decisions:

- docs/transformer-demo-spec.md
- docs/Attention-Is-All-You-Need.pdf
- docs/course-syllabus.pdf

The paper is the source of truth for Transformer mathematics.

## Goal

Build a dark, highly polished mathematical animation that teaches
self-attention through motion and real numerical calculations.

This is a live 5–10 minute Demo Presenter artifact, not a generic website
or dashboard.

## Core requirements

- Real Q/K/V calculations
- Real dot products
- Real scaling by sqrt(d_k)
- Real softmax
- Real weighted sum of Values
- Real multi-head calculation
- Tiny pedagogical dimensions
- Smooth SVG/network-style animations
- Very little prose
- Guided Previous / Next presentation flow
- Explore mode after guided presentation
- Deterministic calculations
- No fake numbers disconnected from internal state
- Dark mathematical-explainer aesthetic

## Development rules

1. Read the full specification before implementing.
2. Do not simplify away mathematical calculations.
3. Preserve object continuity across animations.
4. Avoid dashboard/card-heavy UI.
5. Test every displayed calculation against JavaScript state.
6. Keep the demo reliable when opened in Chrome.
7. Prefer a self-contained index.html unless there is a strong technical
   reason to split the project.
8. When changing mathematical behavior, verify it against the paper.
9. Do not claim pedagogical toy weights are trained Transformer weights.
10. Do not add features just because they look impressive.

## Working process

For significant changes:

1. Explain what you intend to change.
2. Implement it.
3. Test the interaction.
4. Verify the numerical calculations.
5. Check for visual regressions.