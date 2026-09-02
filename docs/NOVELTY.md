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

## 1. The one genuinely novel thing: a reward-integrity guard against non-physical designs

### The problem, stated precisely
In RL circuit sizing, the agent maximizes a reward computed from a SPICE result. But **a
broken circuit can return a perfectly plausible-looking SPICE result.** Concrete failure
(we can reproduce it): push the load resistor high enough that the drain node collapses
below the transistor's saturation voltage. The amplifier is **dead** — output pinned, devices
in triode — yet `.ac` still returns a finite gain (say −39 dB) and `.noise` returns a finite
number. Nothing errors. The reward function sees "ordinary low-gain amplifier," not "corpse."

An RL agent will **find and exploit these regions** — it's the textbook definition of
**reward hacking**: optimizing a proxy while the true objective (a *functional* circuit) is
violated. This is a documented, named failure mode in physical RL — e.g. a fluid-control
policy that "reported drag reduction while collapsing to non-physical configurations"
(arXiv 2606.06227). Our audit found our *own* agent doing exactly this: railed parameters,
an attenuating "equalizer," a design sitting on a boundary that only looks valid.

### Why this is a real gap (this is the defensible part)
The crowded RL-sizing literature does **not** guard against *functional invalidity*:
- **They assume the simulator result is meaningful.** AutoCkt, DNN-Opt, PPAAS optimize on
  the returned metrics directly.
- **GLOVA (2025) does early failure detection — but a *different* failure class.** GLOVA's
  μ-σ / simulation-reordering trick halts *spec* verification early when a design is going to
  *miss specs across corners*. That's "this design is bad." It is **not** "this design is
  **non-functional but scores fine**." Those are different problems: GLOVA saves time on
  designs that visibly fail; we catch designs that *invisibly* pass.
- Classical flows have "DC bias verification" as a manual sign-off step — but nobody has put
  it **inside the RL reward loop as a hard gate that converts reward-hacking exploits into
  rejections.**

**That combination — an operating-point *validity* guard, integrated into the RL reward as a
reward-hacking defense, for high-speed equalizer sizing — is, as far as the 2024–25 literature
shows, unclaimed.** It also happens to be the kind of "boring but real" contribution that a
*professional* judge (who has debugged dead SPICE decks at 2am) will immediately respect,
precisely because the flashy stuff (RL, LLM, PVT) is already taken.

### What the guard actually checks (make it concrete)
For every candidate, **before** trusting the reward, run a cheap `.op` and assert:
1. **Every transistor is in saturation** — `Vds > Vds,sat` and `Vgs > Vth` for each device
   (read `region`/`vdsat` from the SPICE operating point).
2. **No collapsed/railed nodes** — output common-mode inside `[Vov, VDD−Vov]`, not pinned.
3. **Bias sanity** — the mirror actually delivers ~the intended current (mirror error bounded).
4. **Convergence & finiteness** — the op point converged and metrics are finite/non-degenerate.

If any check fails → the design is **rejected** (large negative reward, flagged invalid),
and — the efficiency win — we **skip the expensive characterization** (transient HD3, eye,
noise) entirely. A dead design never consumes the costly sims.

### Why it makes the project *better* (three measurable payoffs)
1. **Reward integrity / trust.** The agent can no longer converge to a phantom. Every design
   it proposes is a *functioning* circuit. This is the correctness argument.
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

### The experiment that proves it (build this — it's the money slide)
- **A/B run:** identical RL setup, guard **OFF** vs **ON**, same seeds/budget.
- **Show the exploit:** with guard OFF, exhibit a design the agent converged to that *passes
  the numeric reward but is non-functional* (transistor in triode, dead amp). This is your
  jaw-drop moment — "here is the agent cheating."
- **Show the fix:** with guard ON, that region is rejected; the agent is forced to
  functionally-valid designs.
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
- **Resident-simulator engineering (~90× faster evals).** This is *engineering, not AI* — so
  **frame it honestly** as "the enabler that made RL-on-real-SPICE feasible in an afternoon,"
  never as the headline result. (The old "350×" claim was measuring simulator startup — do not
  repeat it.)

---

## 3. The honest headline numbers (use these exact framings)

- **Sample efficiency, per spec:** trained policy reaches a full-8-spec design in a **median
  of ~6 SPICE evaluations**, vs **~24–25 for CMA-ES / Bayesian / random from scratch**
  (measured, identical success test). → *"~4× fewer simulations per spec than the strongest
  conventional optimizer."*
- **vs brute force (what Astera actually asked to beat):** an exhaustive parameter sweep is
  ~10⁶+ simulations in 6 dimensions; the policy uses single digits. → the honest "orders of
  magnitude vs sweeping" claim.
- **Amortization (be upfront):** RL carries a one-time training cost (~11k sims), so it only
  *net* pays back after many specs. **Show the break-even curve** — that honesty is itself
  impressive and pre-empts the judge's obvious question. Do **not** hide the training cost.
- **PVT:** full 45-corner sign-off **with a real bias mirror** and the real eye at every corner.

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

> "RL for analog sizing isn't new — AutoCkt did it, and 2025 already has PVT-aware RL and LLM
> agents. So we didn't try to out-RL them. We went after a failure mode *none* of them
> address: **an RL agent will reward-hack a SPICE reward, because a broken, non-functional
> circuit still returns a plausible number.** We built an **operating-point validity guard**
> that sits inside the reward loop, proves every device is in saturation before the metric is
> trusted, and rejects the phantom designs the agent would otherwise exploit. It does two things
> a judge cares about: it makes every proposed design a *real* circuit, and — because it kills
> invalid candidates with one cheap operating-point check instead of a full characterization —
> it **cuts the wasted-simulation budget**, which is the #1 cost in this whole field. Here's the
> agent cheating with the guard off; here's it forced honest with the guard on; here's the
> simulation time we saved."

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
- Reward hacking as a named failure mode in physical RL — arXiv 2606.06227; overview: Wikipedia "Reward hacking".
- Industry pain / simulation bottleneck framing — Synopsys "AI-powered analog design";
  Berkeley BAIR "Sample-Efficient EA for Analog" (simulation cost).
- gm/ID methodology — inversion-coefficient sizing literature.
