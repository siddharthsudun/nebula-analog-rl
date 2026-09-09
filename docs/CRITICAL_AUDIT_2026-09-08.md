# Critical audit — 2026-09-08

**Scope:** documentation corrections only. The baseline location is `work/critical-fixes`.
Historical result artifacts and source code were not changed by this audit.

## Current status

- `docs/PASS_VS_VALID.md` now treats `results/pass_vs_valid.json` as a historical artifact.
  A concurrent `eye.py` self-labelling correction changes future scores. The archived artifact
  remains unchanged; fresh revalidation is pending before its labels or rates are reported.
- `docs/RESULTS_G32_SELFCAL.md` preserves the H-SC development record but no longer presents it
  as a fresh delivery result. The test used a development slice, had fallback cases, and its
  peak-axis observation is constrained by AC-grid quantization. The frozen configuration is the
  delivered configuration; fresh revalidation is pending.
- `docs/NOVELTY.md` no longer claims priority, universal functionality, or unproven metric
  improvements. Headline figures require artifact provenance and corrected-score revalidation.

## Interpretation rules

- Operating-point validity is topology- and device-role-specific. A saturation criterion fits a
  device intended to provide saturated gain; it is not a universal rule for every MOSFET.
- DC attenuation is not universally invalid for a CTLE. Its acceptability depends on the
  topology, receive-chain gain budget, and stated design intent.
- A guard can demonstrate conformance to declared criteria. It does not by itself prove a
  circuit is functional in every relevant operating condition.

## Related-work comparison set

- SelfCal, arXiv:2604.07387: equation-based LLM reasoning, local-slope comparison, and triode
  flags.
- Lighthouse, arXiv:2607.14008.
- FD-MAGRPO, AAAI article 39388: operating-region observations.
- HR2, DOI:10.1109/MLCAD65511.2025.11189231.
- PVT Sizing, DOI:10.1145/3649329.3661850.
- RoSE, arXiv:2407.19150.
- SABLE, arXiv:2607.03701.

These references define the work that must be compared before a novelty or priority claim is
made. They do not by themselves establish SILQ performance or uniqueness.

## Required next evidence

1. Re-run pass-versus-valid with the corrected eye scorer while retaining the historical
   artifact unchanged.
2. Revalidate the delivered frozen configuration and the self-calibrating path with the same
   corrected scoring contract.
3. Document device roles, operating-region criteria, and the link gain budget before classifying
   any candidate as invalid.
4. Attach reproducible artifact provenance to every headline metric before external use.
