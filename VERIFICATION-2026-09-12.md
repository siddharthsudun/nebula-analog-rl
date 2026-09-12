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
| `pytest tests/` | 812 passed, 3 skipped, 2 xfailed | **885 passed**, 3 skipped, 2 xfailed |
| `node tests/*.cjs` | 19 passed, 0 failed | **30 passed**, 0 failed |

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
attributed to the user, and marked as met. The rail stays where it is — that range is the problem statement's, and the tool should
not accept asks outside the scope it claims. So an out-of-range ask is now an **error**:

* the reader strip shows **"Out of range — Peak frequency band"**, naming what you asked
  for, what the control spans, *why* it spans that, and what the run would be scored at
  instead;
* **the run is refused** before any simulator budget is spent, the same way a bad
  signal/noise input is already refused;
* the only way past it is to reword the request, or click **"run it at 2.45–2.50 GHz"** —
  an explicit, per-request consent that the next keystroke clears.

A request that fits every rail still reports *nothing* — a note on an honoured run would
train people to ignore the notes, which is how the silent clamp survived this long.

## 6. A second bug, found while testing the first

Typing a *second* request in the same session silently truncated it.

The two band edges were applied one at a time, and each read the **other knob's current
value** as its bound. So after any ask that parked the knobs at the top of the rail:

> "PCIe Gen2 CTLE, 9 dB boost over a 12 dB channel, **peak between 1.5 and 2.0 GHz**"
> → the band landed on **1.50–2.50 GHz**

which is neither request. Nothing reported it, because each edge *on its own* was inside
the rail. Both edges are now applied as one ask, and "clamped" now means *outside the
rail* and nothing else — an edge the other knob pushed was being reported as a range the
tool cannot represent, and an error that cries wolf on a request the rail can serve is how
a real out-of-range message gets ignored. Three regression tests.

Same class of fix: the band is **one** ask on two knobs, so it now produces one chip
message, one assumption note and one error, not two of each.

## 7. A third bug, in the same corner

Probing one-sided asks after the fix above found the point rule swallowing them:

| ask | before | after |
|---|---|---|
| "peak **below** 3 GHz" | 2.70–3.30 GHz | 1.25–3.00 GHz |
| "peak **at least** 1.8 GHz" | 1.62–1.98 GHz | 1.80–2.50 GHz |
| "keep the peak **under** 2.2 GHz" | 1.98–2.42 GHz | 1.25–2.20 GHz |

"Peak below 3 GHz" was read as a band *centred* on 3 GHz — inventing a 2.7 GHz **floor**
the user never stated and a 3.3 GHz ceiling **above** the one they did. The run was then
scored against a band that contradicts the request on both sides and reported it
satisfied. Eighteen comparator spellings now bound one edge and leave the other at the
competition default, and say so in `assumptions`. Twenty new tests.

One more guard came out of the same probe: **"8 dB peaking, 5 GHz baud rate"** was read as
a request to move the peak to 5 GHz. The rate words were guarded only *before* the figure,
not after it, so a link speed quoted after the peaking figure planted a 4.5–5.5 GHz band —
outside the tunable range entirely, steering the search at something the circuit cannot do
for a requirement nobody made.

Across every way I could think of to state the peak — two-sided, point, one-sided, and the
eight traps that must claim nothing — the parser now reads **44/44**.

Two consequences fixed with it: the unstated edge was keeping the *previous* request's
value ("peak below 2.2 GHz" then "peak at least 1.8 GHz" landed on 1.80–2.20 GHz, which
makes the assumption sentence a lie), and the error message printed a one-sided ask as a
fake span — "you asked for 3.00–2.50 GHz" made the user's own request look like nonsense.

## 8. Does a band above the rail actually solve?

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

## 9. Decision taken

**The rail stays at 1.25–2.50 GHz.** The circuit and the API can both go above it —
2.70–3.30 GHz solves in 10.6 s and passes the guards — but that range is the problem
statement's tunable range, and a tool that quietly accepts asks outside the scope it
claims is worth less than one that says no. An out-of-range ask is therefore an error
that stops the run, not a rounding (section 5).

## 10. Flags

`BUS_KEY_LOCATION.txt` (1,396 bytes) sits untracked in the repo root and is **not
gitignored**. I have not opened it. Given the name, it is exactly the file the
"never `git add -A`" rule exists to protect — worth either ignoring it or moving it out
of the tree.

**The requirement panel is sticky, and I have not changed that.** A value applied from
one request stays in its box when the next request does not mention it: type "under 12 mW"
and then a request with no power ceiling, and the run still carries 12 mW. It is visible
in the panel — unlike the band collision in section 6, which produced a number neither
request contained — so it is defensible as a deliberate design. But it is the same family
of surprise, and whether the panel should reset per request is a product call rather than
a defect I should silently decide.
