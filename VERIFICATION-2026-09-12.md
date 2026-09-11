# Full-codebase verification, 12 Sep 2026

Scope asked for: run the whole suite; test every parameter including SNR and Pareto;
change the hard limits; change the input sliders; make sure the LLM recognises every
possible input, and check whether peak frequency reached it.

Nothing in the frozen architecture moved. Every change is in the parser, the dashboard,
or the tests — the three areas the 26 Aug freeze leaves open. The competition-default
path is unchanged and measured to be unchanged (control run at the bottom).

---

## 1. The suite

| Suite | Before | After |
|---|---|---|
| `pytest tests/` | 812 passed, 3 skipped, 2 xfailed | **865 passed**, 3 skipped, 2 xfailed |
| `node tests/*.cjs` | 19 passed, 0 failed | **22 passed**, 0 failed |

The 3 skips are the ngspice-not-installed DFE tests. `pdk.available()` is true on this
machine, so the seven Pareto tests CI has to skip **do run here, and pass** — the
`needs_pdk` markers cost nothing locally.

CI on `0d509110c` is green on both jobs. That was the first all-green run on
`feat/silq-overhaul`; it closes the thread the node-id deselects opened.

## 2. "I think the peak frequency is not added to the LLM yet"

Half right, and the half that was wrong was the more dangerous half.

The field *was* wired: `peak_freq_lo_ghz`/`hi` are in the LLM system prompt, in the
plausibility box, and in `REQUIREMENT_FIELDS`, so a parsed band does steer the search.
What did not exist was **any test of it** and **any rule for the natural way to say it**.

Measured before the change, on 37 phrasings a real engineer could type: **19/37**.
Every failure was the peak band. Measured after: **37/37**.

What was broken:

* **A point ask read as nothing.** "peak at 3 GHz", "peak frequency of 2.4 GHz", "put
  the peak at 4 GHz", "centre the peak on 3.5 GHz", "fpeak 2.2 GHz" — the only rule was
  a *range* rule, so all of these set no band at all and the run silently used the
  competition band. A point ask now becomes a declared ±10% band, and the width is
  reported as an assumption the user can see and override.
* **±10% is measured, not chosen by taste.** Against the 374,588-row corpus it leaves
  between 1,879 and 14,299 candidate designs everywhere from 1.5 to 5 GHz; ±5% leaves as
  few as 759. The corpus is the only start generator that can see a requested band, so
  that count is what decides whether the ask is reachable at all.
* **An inverted range passed straight through.** "peak between 2.5 and 2 GHz" produced
  `lo=2.5, hi=2.0`, which reaches `hard_pass` as a `peak_in_band` check no design can
  satisfy — the run then reports failing a requirement the user never set. Now sorted.
* **MHz was not a unit.** "peak at 2400 MHz" read as nothing. Now 2.4 GHz.
* **Nyquist could be mistaken for the peak.** Guarded both ways: "9 dB of peaking for a
  5 GHz Nyquist link" and "9 dB peaking at 4 GHz Nyquist" and "4 GHz of bandwidth" all
  now refuse to set a band. Eight such traps are pinned.
* **"5 GHz Nyquist" (number first) set no Nyquist at all** — a pre-existing gap in a
  different field, found while guarding the one above. Fixed.

The LLM prompt now carries the same point-ask convention as the rules, so the two
readers agree instead of arriving as a conflict the user has to arbitrate.

## 3. SNR

**21/22 phrasings** before, **22/22** after. The gap: `INTENT` accepted
"signal-to-noise" but the value rules anchored on the three letters `snr`, so
*"signal-to-noise ratio of 25 dB"* switched SNR **on** and dropped the 25 — the user then
saw an SNR panel demanding a measured input SNR they had just typed. The vocabulary is
now shared, and the *refusal* was widened with it: an output-SNR target said in words is
refused exactly like `output SNR`, because executing one as a measured input SNR scores
the run against a number nobody measured.

## 4. Hard limits

All eleven settable limits, moved tighter, followed to the verdict: **11/11** apply,
label their direction correctly, and actually bind. "Bind" is the one that matters — a
limit can be stored and labelled and never read, and then the interface shows a tightened
bar while the run passes anyway. Each is now checked with a measurement sitting *between*
the default and the new limit, so it must pass at the default and fail after.

This was previously covered for one field; it is now a per-field matrix, plus a test that
fails if a field is added to `REQUIREMENT_FIELDS` without a row in it.

## 5. Input sliders — the worst thing found

**The dashboard was silently rewriting the user's request.**

`applyParse` clamped every parsed value into its slider's rail and said nothing. The
rails are narrow on purpose — target boost 3–12 dB, channel loss 6–20 dB, the peak band
one octave, all from the problem statement — so clamping is defensible. Doing it in
silence is not. Measured:

> "20 dB of boost, peak at 3 GHz" → the band knobs land on **2.45–2.50 GHz** and
> `peak_freq_lo_ghz: 2.45` is sent to the API **as the user's own requirement**, boost is
> quietly cut to 12 dB, and the run then reports both satisfied.

2.45–2.50 GHz is not a rounded version of 3 GHz — it is a different requirement,
attributed to the user, and marked as met. `applyParse` now returns every ask its rails
could not represent and the composer prints it at refusal severity, naming the ask, the
rail and what the run will actually be scored at. Three new tests pin it, including one
that a request which fits every rail reports *nothing* — a note on an honoured run would
train people to ignore the notes.

## 6. Does the widened band actually solve?

The API has no rail, so a band above 2.5 GHz is now reachable for the first time.
Measured, `thinking` mode, 8 dB target over a 12 dB channel:

| Ask | Band | Time | Result | Measured peak | Starts |
|---|---|---|---|---|---|
| "peak at 3 GHz" | 2.70–3.30 | **10.6 s** | solved, guard-valid | 3.193 GHz | 0 PPO, 1 corpus |
| "peak at 2.4 GHz" | 2.16–2.64 | 39.1 s | solved, guard-valid | 2.393 GHz | 4 PPO, 4 corpus |
| "peak at 1.8 GHz" | 1.62–1.98 | **1.1 s** | solved, guard-valid | 1.694 GHz | 0 PPO, 1 corpus |
| *control: no ask* | 1.25–2.50 | 17.0 s | solved, guard-valid | 1.901 GHz | 2 PPO, 0 corpus |

The control reproduces the frozen behaviour exactly — winner `ppo:4001`, two restarts,
`n_surrogate_starts` 0 — so the band-first reordering still does not fire at the defaults.

## 7. One decision that is yours, not mine

**Should the peak-band rail be widened past 2.5 GHz?**

The circuit can do it: 2.70–3.30 GHz solves in 10.6 s and passes the guards. The parser
and the API can both express it. The *only* thing stopping a user is the slider, whose
comment ties its range to the competition problem statement's tunable range.

Widening it would let the tool accept asks outside the competition's stated scope. That
is a scope-and-credibility call, so I have not made it. Until you do, an out-of-rail ask
is now clamped *visibly* instead of silently, which is correct either way.

## 8. Flag

`BUS_KEY_LOCATION.txt` (1,396 bytes) sits untracked in the repo root and is **not
gitignored**. I have not opened it. Given the name, it is exactly the file the
"never `git add -A`" rule exists to protect — worth either ignoring it or moving it out
of the tree.
