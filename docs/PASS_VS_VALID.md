# Historical artifact: published specs alone do not establish intended operation

**Status:** historical artifact awaiting fresh revalidation. `results/pass_vs_valid.json` is
unchanged. A concurrent correction to `eye.py`'s self-labelling changes future scores, so this
document does not treat the archived counts as a current result.
**What it is about:** the problem, not our solver.

---

## 1. The claim

Take the eight hard specs the problem statement publishes. Search for designs that pass all
eight. Then ask a separate question — does each one meet the documented operating regime and
link-level intent?

**The archived run labelled 24 of 28 eight-spec passers as rejected by its guard layer.** These
historical labels must be revalidated with the corrected eye scorer before they are quoted.

That number splits into two very different tiers, and the split matters more than the
headline:

| tier | count | share of the 28 | can this be argued with? |
|---|---|---|---|
| `T2.5_mosfet_not_in_saturation` | 10 | 36% | Requires device-role review; this conflicts with an intended saturated gain device, not every MOSFET topology. |
| `T4.10_dc_gain_implausible` (DC gain < 0 dB) | 14 | 50% | Context-dependent; DC attenuation can be intentional in a CTLE and link gain budget. |
| **combined** | **24** | **86%** | |

Neither percentage is a current validity rate. Fresh revalidation must first confirm the
historical labels, then apply topology-specific device-role and link-budget criteria.

The durable question is:

> Do the published checks state the topology-specific operating region and link-gain intent
> needed to distinguish a desired equalizer configuration from an unintended one?

That is a statement about the specification, not about anybody's optimizer.

---

## 2. Why the published checks may be insufficient

The eight published checks (`hard_pass`, `src/eqrl/specs.py:97`) are:

`boost_range`, `peak_in_band`, `hd3`, `noise`, `power`, `area`, `eye_h`, `eye_v`

> **Eight is a deliberate setting, and it is not our default.** Since `2f3ec52b3`
> (18 Aug 2026) `DEFAULT_SPEC.dc_gain_db_min = 0.0`, so `hard_pass` against our *own* default
> spec emits **nine** checks. The measurement below scores against the **competition's**
> published set — `dc_gain_db_min=None`, pinned explicitly in `spec_pass`
> (`src/eqrl/experiments/pass_vs_valid.py:76`). That is the correct experimental design: the
> question is whether designs meeting the *published* spec are real circuits, so the
> published spec is what they must be scored against.
>
> Three independent confirmations that the shipped artifact is an eight-check artifact:
> 1. **Dates.** `pass_vs_valid.py` and `results/pass_vs_valid.json` were both committed in
>    `6d465c93a` (17 Aug 2026), when `dc_gain_db_min` was still `None`. The flip came a day
>    later, and the script has exactly one commit in its history.
> 2. **Internal evidence.** 14 of the 28 designs fail `T4.10_dc_gain_implausible`
>    (DC gain < 0 dB) — impossible for a design that had also cleared a `dc_gain ≥ 0 dB`
>    hard check. The artifact proves its own check count.
> 3. **Contract.** `spec_pass`'s first docstring line is "Exactly what `honest_benchmark`
>    scores on", and `honest_benchmark._DC_GAIN_DB_MIN` defaults to `None` for the same
>    reason.
>
> The pin is recent, and the reason matters more than the fix. Until 07 Sep 2026 the spec was
> built by `dataclasses.replace(DEFAULT_SPEC, ...)` without naming the field, so it silently
> **inherited** the 18 Aug flip: re-running the script produced nine checks and a different
> split from the artifact it shipped with, while this document cited eight. Nothing in the
> repo distinguished which was right. That is a defect in reproducibility, not in the result
> — but it is the exact failure this document is about, committed by us, one directory over.

**There is no DC gain check among them.** That omission deserves a documented design-policy
decision, because boost is defined as a difference:
boost is defined as a *difference*:

```
boost_db = peak_db − dc_db
```

A search can raise `boost_db` by lifting the peak or by reducing DC response. A source-
degenerated CTLE may intentionally attenuate at DC relative to its peak, so DC attenuation is
not universally invalid. Its acceptability depends on the topology, system gain budget, and
the intended role of the CTLE in the receive chain.

The boost and peak checks use that same difference. Depending on the surrounding link model,
other checks can make attenuation appear attractive. This is a testable specification risk,
not proof that all DC attenuation is an exploit.

An optimizer follows the supplied objective. The specification should therefore state its
operating-region and link-gain assumptions explicitly.

---

## 3. The measurement

| | |
|---|---|
| Raw data | `results/pass_vs_valid.json` |
| Search method | CMA-ES |
| Specs | 4 |
| Budget | 60 evaluations per spec (240 total) |
| Archived eight-spec passers | 28 |
| Archived guard rejections | 24 |
| Archived breakdown | 14 × `T4.10_dc_gain_implausible`, 10 × `T2.5_mosfet_not_in_saturation` |

The four specs, as (target boost over channel loss):

| target boost | channel loss | margin |
|---|---|---|
| 8.82 dB | 10.16 dB | +1.34 dB |
| 5.25 dB | 8.13 dB | +2.89 dB |
| 9.88 dB | 15.30 dB | +5.42 dB |
| 8.64 dB | 13.84 dB | +5.20 dB |

This archived count is not a solver-performance comparison. It is also not a current property
of the specification until the corrected eye scoring path has revalidated it.

---

## 4. Why it matters for the competition

An entry that reports only published-check passes does not, by itself, document the intended
operating regime or link-gain behavior. The archived labels here are not evidence of a current
majority statement until revalidated.

That has three consequences:

1. **A headline solve count needs operating assumptions** when it is presented as circuit
   evidence.
2. **A validity guard must be topology-specific.** A global saturation or DC-gain rule can
   reject a legitimate circuit role.
3. **A DC-gain floor is a design policy, not a universal CTLE law.** It should be justified by
   the stated topology and end-to-end link gain budget.

---

## 5. Limitations and required revalidation

State these before anyone else does.

- **The historical labels are pending revalidation.** The `eye.py` self-labelling correction
  changes future scores. The archived file remains unchanged for traceability; it must not be
  presented as a fresh result.
- **All four specs are positive-margin** (channel loss exceeds target boost, by +1.3 to
  +5.4 dB). The negative-margin regime — where requested boost exceeds channel loss, and
  where our own policy scores 0% — is **not represented here at all.** We do not know
  whether the effect is stronger or weaker there, and should not imply we do.
- **Both guard categories require design intent.** DC attenuation can be valid, and saturation
  checks apply only to devices expected to amplify in saturation. Node, device-role, and link
  criteria must be stated before classifying a candidate.
- **These are SILQ guard definitions.** They need independent review and a fresh rerun before
  they support a comparative or functionality claim.

---

## 6. How to refute this

Stated so a reader can check rather than take our word:

1. Re-run the archived experiment with the corrected eye scorer and preserve the fresh result
   beside the untouched historical artifact.
2. State each device role and the intended operating region before applying a guard.
3. Include link gain-budget evidence when evaluating DC attenuation.
4. Re-run across a broader specification set, including negative-margin conditions.

---

## 7. What we did about it

- A DC-gain constraint may be used when it is justified by the intended topology and link
  budget; it must not be presented as a universal invalidity rule.
- The delivered result remains the historical delivery. Fresh delivered-design revalidation
  with the corrected eye scorer is pending; historical artifacts remain unchanged.
