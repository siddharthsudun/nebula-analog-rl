# Task 9 — corner screening within G3.2, design only

No implementation, retraining, re-optimization of the delivered circuit, or run is
authorized here. This is a separately named future `g32_corner_screen_v1` experiment;
the frozen G3.2, PPO, guards, rewards and historical artifacts remain unchanged.

## Proposed mechanism

Wrap the candidate-evaluation/acceptance interface of a separately named G3.2
refinement arm. A nominally eligible proposed step spends a small additional corner
budget before it can replace the incumbent. A screen failure rejects this proposed
step in the new arm; it is not a modified reward or a relaxed nominal constraint.
Maintain separate records for nominal acceptance and screen acceptance. The frozen
baseline receives the same total actual simulator-call ceiling, so screen calls
compete with nominal refinement calls. Do not call an unmeasured corner a pass.

Initial screen candidates, motivated by existing failure modes, not claimed worst
for every design:

1. SS / 1.71 V / 125 C: saturation/headroom stress.
2. FS / 1.71 V / 125 C: delivered worst target-error corner.
3. SS / 1.89 V / 125 C: delivered worst DC-gain corner.

The third matters: `delivered_circuit.json#/pvt/worst_case_by_metric/dc_gain_db`
records minimum **1.0404810884607727 dB**, at **ss|1.890|125**, against the additional
0 dB floor. “All worst corners are low supply” would be false. DC-gain-floor failures
are requirement violations; saturation failures concern the intended device roles.
Report them separately. These archived numbers retain their legacy-eye scope.

Start with full `fast=False` guarded candidate evaluations at screened corners.
Three corners are already a cheap approximation relative to 45. An even cheaper
OP/AC-only screen must be a distinct diagnostic API that cannot claim measured noise,
HD3 or a complete guard verdict; never masquerade fast=True placeholders as full
corner verification. No threshold changes or surrogate predictions count as passes.

Bind each record to design fingerprint, target/channel, process/V/T, instrument
version and source hash. Cached screen results are reusable only for exactly that
key. Switching process decks has real cost; count reload analyses/time and use
isolated batches only where this does not change candidate acceptance order.

## Validation and separation

Existing 22 candidates are **development evidence only**. Their 1/22 full-grid
success rate cannot validate a screen designed from their failure patterns. Do not
reuse the delivered design or mode-audit final candidates to select corners/rules.

Freeze the three-corner rule on development data, then collect an independent
validation cohort of up to **100 nominally eligible designs**, with at most **2,000
proposal evaluations** and **4,500 full-grid corner evaluations**. Stop at the
first cap reached; report an insufficient cohort if 100 are not found. Perform
every validation design's complete 45-corner sweep, including those the screen
rejects, so false rejects and missed failures can both be counted. Obtain screen
labels by selecting the frozen three rows of this full measurement, avoiding a
second SPICE charge solely for validation labels.

Report both denominators:

- failure among screen-pass designs = screen-pass AND full-grid-fail / screen-pass;
- missed-failure sensitivity complement = screen-pass AND full-grid-fail / full-grid-fail.

Use one-sided exact binomial 95% upper bounds, plus counts and interval assumptions.
Candidate trajectories can be correlated; cluster by independently initialized
search and report that limitation. For iid designs, zero misses among 59 screen-pass
designs gives an upper bound 1 - 0.05^(1/59), about 4.95%. Zero among fewer does
**not** establish a <5% bound. A 100-design cap does not guarantee 59 screen passes;
inconclusive means do not promote. This gate controls residual risk, not a proof
that the screen predicts all 45 corners.

Only after a passing validation gate, register a separate final comparison using
new specs/designs and a disjoint RNG stream. Compare frozen G3.2 versus the named
screened arm under equal actual measure_all/analysis ceilings. Freeze seeds/budgets
before any final candidate generation. Every final returned candidate still pays
for all **45 corners**; none is declared robust from the screen alone. No final
failure may trigger corner-set tuning followed by reusing that final set.

Primary future outcome: full-grid passing designs / all requested specs. Secondary:
nominal target error, missed failures, screen rejects, total simulator calls,
process reload cost, no-delivery count. A better nominal median with fewer verified
deliveries is not a robustness improvement. HR2's Monte Carlo failure ratio is a
different quantity and is not a comparator for this fixed-grid rate.
