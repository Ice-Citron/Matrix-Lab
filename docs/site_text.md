<!-- Landing-page text for analysis/site.py, from README.md at b0d3015. The README no longer carries these sections. -->

**Find where your benchmark stopped before the hidden eval does.** [Live page](https://ice-citron.github.io/gaphunter/)

*`hole_pose()` error is a modelled r³ distortion mirroring the AIC calibration radius, not a measured camera effect.*

## Problem

In Intrinsic AIC 2026 our policy scored 141 locally and 89 on the hidden evaluation.
The local scenes randomised the board over a radius a third smaller than the real one.
The benchmark never tested where the policy fails.

## What GapHunter does

**Sample → run → map → report.** GapHunter samples the hole offset (`dx`, `dy`, yaw) from the
benchmark's box and a wider deployment box, runs the policy, and maps where
it fails. Task: MuJoCo UR5e peg-in-hole, 20 mm square peg, 21 mm hole. Success: depth ≥ 30 mm, lateral error < 1 mm.

## Two axioms

1. The goal of any robotics firm is to deploy a system with four-nines reliability as cheaply as possible.
2. Each step-up in the reliability regime — 90%, 99%, 99.9%, 99.99% — is equally difficult.
