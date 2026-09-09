# Three-seed solve-rate audit (SILQ-Forensic-Audit MUST-DO #1)

Protocol: target_audit.py, spec_seed=0 (the frozen historical 32 specs), budget=20,
strict tol +/-1.5 dB. Same protocol the CMA-ES/TPE baselines below use.

- seed1 (results/seq_seed1_40k.zip): loose 23/32, strict 9/32
- seed2 (results/seq_seed2_40k.zip): loose 25/32, strict 12/32
- seed3 (results/seq_seed3_40k.zip): loose 18/32, strict 5/32

**Loose solved -- median 23, range 18-25 (of 32)**
**Strict solved -- median 9, range 5-12 (of 32)**

This is the number the audit says the headline (single-seed) solve rate cannot
stand in for. MUST-DO #1 is now satisfied.

## Search-baseline context (MUST-DO #2, budget=20 -- still short of the RL number)
- CMA-ES @ budget 20: loose 15/32, strict 4/32
- TPE @ budget 20: loose 10/32, strict 7/32

Both are still well under the RL median at the same budget=20 -- this baseline still
needs to be re-run at escalating budget until one of them matches or plateaus, per
the audit's MUST-DO #2. Not attempted in this run (separate, still open).
