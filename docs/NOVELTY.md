# What actually makes SILQ novel — and how to defend it to Astera

*Strategy + evidence doc. Read this before the presentation. Every claim here is sourced;
the goal is that a professional analog/ML judge nods, not squints.*

---

## 0. The uncomfortable truth (say this to yourselves, never to the judge)

Most of what we built is **execution, not novelty**. A judge who reads the field will know:

- **RL for analog sizing** was done by **AutoCkt (Berkeley, DATE 2020)** and **DNN-Opt (2021)**.
- **gm/ID-based RL sizing** exists (2023).
- **PVT-robust RL sizing** is a *2025* topic: **GLOVA** (risk-sensitive RL, DAC 2025) and
  **PPAAS** (PVT- & Pareto-aware, goal-conditioned RL, 2025), plus "duel-play probabilistic
  model" RL for PVT-robust sizing (2025).
- **LLM front-ends** for analog design are everywhere now: **AnaFlow**, **AutoSizer**,
  **LEDRO**, **White-Box Reasoning** (LLM + gm/ID), all 2025.
- **Sample-efficiency** tricks (probabilistic model rollouts, ensemble critics) — 2024/25.

So if we walk in claiming "RL designs analog circuits" or "our RL is PVT-robust" or "we have
an LLM wrapper," the judge has seen all three this year. **We must claim something narrower
and truer.** The good news: there is a real, defensible gap, and we're already sitting on it.

---

## 1. Candidate contribution: a topology-specific reward-integrity guard

### The problem, stated precisely
In RL circuit sizing, the agent maximizes a reward computed from a SPICE result. A simulator
can return finite small-signal metrics for a bias point that is inconsistent with the intended
topology and operating regime. For example, an intended common-source gain device can leave
saturation while `.ac` and `.noise` still return finite values. The reward must therefore carry
explicit assumptions about device roles, node headroom, and the intended link gain budget.

An agent can optimize such a mismatch between a proxy and the intended circuit behavior. That
is a reward-integrity risk, not yet evidence that every low-gain or non-saturating candidate is
non-functional. The historical pass-versus-valid labels require fresh revalidation after the
`eye.py` self-label correction; see `docs/CRITICAL_AUDIT_2026-09-08.md`.

### What can be claimed after comparison
This is a candidate contribution, not a field-wide priority claim. The related work named in
§7 includes methods that reason from equations, inspect operating regions, or address robust
analog optimization. The required comparison is whether SILQ's *specific* combination of
topology-specific operating-region checks, reward gating, and early characterization skipping
differs materially from those methods. Until that comparison and fresh revalidation are
complete, do not claim "first," "nobody," or an unqualified functionality guarantee.

### What the guard actually checks (make it concrete)
For every candidate, **before** trusting the reward, run a cheap `.op` and assert the
topology-specific conditions documented for that circuit:
1. **Operating-region checks by device role.** Require saturation only for devices intended to
   provide saturated gain; a device deliberately used as a switch, resistor, or other
   non-saturating element needs its own criterion.
2. **No collapsed/railed nodes** — output common-mode inside `[Vov, VDD−Vov]`, not pinned.
3. **Bias sanity** — the mirror actually delivers ~the intended current (mirror error bounded).
4. **Convergence & finiteness** — the op point converged and metrics are finite/non-degenerate.

If a documented condition fails, the design can be rejected (large negative reward, flagged
for review) and the expensive characterization can be skipped. The time-saving benefit remains
a hypothesis until measured with the corrected scoring path.

### Why it makes the project *better* (three measurable payoffs)
1. **Reward integrity / trust.** The agent is constrained by explicit, reviewable operating
   assumptions. This is evidence of conformance to those assumptions, not proof of universal
   circuit functionality.
2. **Sample efficiency = the #1 industry pain.** Industry's dominant complaint is
   **simulation cost**: "many learning-based algorithms require thousands of simulated data
   points, impractical for expensive-to-simulate circuits," and "simulation of post-layout
   SerDes equalizers is a major bottleneck of iteration time." Our guard rejects invalid
   candidates with **one cheap `.op`** instead of a full multi-analysis characterization,
   so the wasted-simulation budget drops. **Quantify it:** *"X% of the designs a naive agent
   explores are non-functional; the guard screens them for ~1 cheap op each instead of a full
   ~250 ms characterization, cutting wasted simulation time by Y%."*
3. **PVT realism.** Combined with our **real current-mirror bias** (not an ideal source), the
   guard ensures the agent respects the actual bias/headroom constraints that break real chips.

### The experiment that tests it
- **A/B run:** identical RL setup, guard **OFF** vs **ON**, same seeds/budget.
- **Show the operating-regime mismatch:** with the guard OFF, exhibit a candidate that passes
  the numeric reward while violating a documented device-role or headroom criterion.
- **Show the effect of the guard:** with the guard ON, show whether that region is rejected and
  whether the resulting candidates meet the same documented criteria.
- **Two numbers:** (a) *invalid-design rate* the guard caught; (b) *simulation time saved* by
  early rejection. Plot both. This converts "we have a guard" into evidence.

---

## 2. Supporting differentiators (real, but secondary — lead with the guard)

These aren't unique on their own, but stacked with the guard they make a coherent
"physically-honest RL equalizer designer" story:

- **Eye-in-the-loop optimization.** Most sizing papers optimize **AC proxies** (gain, BW,
  phase margin). We optimize the **actual eye** (height/width) through a real channel + adapted
  1-tap DFE — the BER-relevant *system* metric. "We close the loop on the end metric, not a
  proxy" is a legitimate, application-grounded angle for a *SerDes* equalizer specifically.
- **Channel-adaptive design.** The agent conditions on the channel loss and adapts the peaking
  — closer to how a real link must work across cables/boards — rather than hitting one fixed
  spec. (Addresses the "only one scalar varies" critique.)
- **Real current-mirror bias.** We model the bias network that actually drifts across PVT, so
  our robustness numbers mean something. Many demos quietly use ideal sources.
- **Resident-simulator engineering.** This is engineering, not AI. Its performance benefit is
  pending provenance; do not use speedup figures as a headline.

---

## 3. Headline metrics are pending provenance

Do not quote evaluation counts, speedups, training cost, break-even, or PVT-coverage figures
from this document. Each needs a cited artifact, an explicit scoring definition, and fresh
revalidation after the `eye.py` correction. The audit record is
`docs/CRITICAL_AUDIT_2026-09-08.md`.

---

## 4. Industry pain points we can name (grounds the pitch)

| Pain (sourced) | How SILQ speaks to it |
|---|---|
| "Analog workflows remain labor-intensive, expert-dependent" — the automation gap | RL sizes with zero human tuning |
| "Simulation cost is the bottleneck; thousands of sims impractical" | guard skips full sims on invalid designs; RL amortizes |
| "Post-layout SerDes equalizer simulation is a major iteration bottleneck" | we target exactly this block; eye-in-loop |
| "Optimize across hundreds of PVT corners" (Synopsys framing) | 45-corner sign-off with real bias |
| Reward hacking / non-physical optima in physical RL | the guard is a direct, named defense |

---

## 5. The pitch — sentences to actually say to Astera

> "RL for analog sizing is established work. Our contribution under evaluation is a
> topology-specific operating-point guard inside the reward loop. It makes the operating
> assumptions explicit before the reward is trusted, and it can reject candidates that violate
> those assumptions before expensive characterization. We will compare this behavior directly
> with related methods and show a guard-off/guard-on revalidation, rather than claiming that it
> proves functionality or has no precedent."

Then show: the exploit design, the A/B curves, the 45-corner sign-off on the real-bias circuit,
the eye opening, and the honest sample-efficiency + break-even numbers.

---

## 6. Two-week plan to make it bulletproof

**Must-do (turns the claim into evidence):**
1. **Implement the guard** as a hard pre-reward gate (op-point region checks + node/bias sanity).
   ~1–2 days. Wire it so invalid → reject + skip full characterization.
2. **The A/B experiment** (guard off vs on): invalid-rate caught, sim-time saved, plus an
   exhibited exploit design. ~1 day. *This is the presentation centerpiece.*
3. **Re-run sample-efficiency vs CMA-ES** with the guard on, 3–5 seeds, and the honest
   **break-even curve**. (Code already exists: `honest_benchmark.py`.) ~1 day.

**High-value:**
4. Randomize the channel in training (already supported) → strengthen the channel-adaptive story.
5. **gm/ID parameterization** — size by inversion level so every sample lands on a valid bias
   point *by construction*. This both boosts sample efficiency and makes the agent's choices
   *interpretable* to an analog judge ("it chose gm/ID ≈ 14, trading headroom for noise"). It
   also pairs beautifully with the guard (guard = safety net, gm/ID = staying in-bounds by
   design). Highest-impact optional add.

**Do NOT spend time on:** more RL hyperparameter tuning, the website chrome, or re-claiming
350× / PVT-RL / LLM novelty. Those are either done or taken.

---

## 7. Sources (for your own confidence + to cite if asked)

- AutoCkt — RL analog sizing (DATE 2020). DNN-Opt — arXiv 2110.00211.
- GLOVA — Global/Local Variation-Aware, risk-sensitive RL — arXiv 2505.11208 (early-failure
  detection is *spec* failure, not functional validity).
- PPAAS — PVT & Pareto-aware goal-conditioned RL — arXiv 2507.17003.
- LLM agents: AnaFlow (arXiv 2511.03697), AutoSizer, LEDRO, White-Box Reasoning (arXiv 2508.13172).
- SelfCal — arXiv:2604.07387 (equation-based LLM reasoning, local-slope comparison, and triode
  flags); Lighthouse — arXiv:2607.14008; FD-MAGRPO — AAAI article 39388 (operating-region
  observations).
- HR2 — DOI:10.1109/MLCAD65511.2025.11189231; PVT Sizing —
  DOI:10.1145/3649329.3661850; RoSE — arXiv:2407.19150; SABLE — arXiv:2607.03701.
- These references define the comparison set; they do not establish a priority claim or a SILQ
  performance result.
