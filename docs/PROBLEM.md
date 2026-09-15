# Problem Statement (Analog Track)

Verbatim from the Nebula @ BITS Goa poster, plus our engineering interpretation.

## As stated

> **AI/ML for Analog Circuit Design.** Demonstrate a fully automated framework for
> schematic design of analog circuits using Reinforcement Learning. Given circuit
> specifications, build an automated design flow that sizes devices using fewer search
> spaces (lowest design time) to reach near-optimal solutions, with zero human
> intervention. Use case: Equalizer circuit design (PCIe PHY). Should take significantly
> lower time than sweeping all MOS, R, C, L parameter space, with zero human
> intervention.

### Specifications
- Nyquist Freq (PCIe Gen 2, 5.0 Gbps): **2.5 GHz**
- HF peaking boost **3–12 dB** (tunable **1.25–2.5 GHz**)
- **1-Stage CTLE w/ source degeneration** (variable **Rs, Cs**) **+ 1-Tap DFE**
- **NRZ** signaling
- Linearity **HD3 < −30 dB** (100 MHz diff input)
- Input-referred noise **< 1.5 mV_rms** (10 MHz – 5 GHz)
- Power **< 15 mW**
- Area **< 0.05 mm²** (130 nm PDK)
- Eye opening **> 0.4 UI (H)** and **> 100 mV (V)**
- Meet across PVT: **TT, SS, FF, SF, FS**; **VDD ±5%**; **0–125 °C**

### Objective
Develop an RL-based automation framework to automatically design an equalizer for the
given specifications, with minimal to zero human intervention.

### Deliverables
1. A Reinforcement-Learning-based **Python framework** that takes target specs as input,
   seamlessly integrates with a SPICE simulator — **outputs the final schematic and
   resulting specs**.
2. **Bonus:** LLM-based human interaction with a wrapper to fine-tune.

### Tools & Resources
Python · Anaconda · LLM · Gym Utils · SPICE simulators (NGSpice or PySpice) ·
Open-source PDK (IHP 130 nm BiCMOS or SkyWater SKY130).

---

## Our interpretation / decisions

### What "equalizer" means here
A receiver equalizer that undoes channel loss at high frequency. Two blocks:
- **CTLE (Continuous-Time Linear Equalizer):** a differential pair with a **source-
  degeneration RC** (Rs ∥ Cs) between the sources. The zero from Rs·Cs creates the HF
  **peaking** that boosts the Nyquist band and flattens the overall response. This is the
  primary analog block we size.
- **1-tap DFE (Decision-Feedback Equalizer):** cancels the first post-cursor ISI tap.
  For the schematic-sizing scope we model it as a summing node + slicer + single feedback
  tap; the "tap weight" is a knob the agent tunes. (Keep this lightweight — the CTLE is
  where the analog sizing action is.)

### Design variables (the RL action space)
Six continuous knobs, exactly as implemented in `ACTION_SPACE` (`src/silq/circuits/ctle.py:124`):

| Variable | Meaning | Range |
|---|---|---|
| W_in | input diff-pair width | 1–100 µm |
| L_in | input diff-pair length | 0.15–1 µm |
| I_tail | tail bias current | 0.05–1.0 mA |
| Rs | degeneration resistor | 100 Ω – 5 kΩ |
| Cs | degeneration cap | 10 fF – 2 pF |
| R_load | load resistor | 100 Ω – 5 kΩ |

**Not action variables, by deliberate choice:**
- **`w_dfe`** (`ctle.py:37`) is a field on `DesignVars` but is pinned at 0.0 and
  excluded from `ACTION_SPACE` (declared at `ctle.py:124`; the exclusion of
  `w_dfe`, and the reason for it, at `ctle.py:97-99`). The 1-tap DFE is a receiver-DSP block, not an
  analog knob: it is applied at the slicer and **adapted at runtime to the measured first
  post-cursor** in the eye engine (`src/silq/sim/eye.py:120-128`). It is always on — the
  eye metric in `measure_all` is post-DFE (`src/silq/sim/measures.py:174`), so every
  reward and all 45 PVT corners are scored with it active. Sizing a tap at design time
  would model a receiver nobody builds; a real DFE adapts to the channel it sees.
- **`C_load`** is a fixed parasitic model, not searched.

Start continuous (PPO/DDPG); optionally discretize per-knob (AutoCkt style) if training
is unstable.

### Measurements (the observation vector)
Extracted from SPICE per candidate — see `src/silq/sim/measures.py`:
- **DC/AC:** peaking (dB), peak frequency, DC gain, boost = peak − DC
- **Linearity:** HD3 from a transient/PSS at 100 MHz diff input (FFT of output)
- **Noise:** integrated input-referred noise 10 MHz–5 GHz (`.noise`)
- **Power:** VDD · I_total from operating point
- **Area:** analytic estimate from device geometry (ΣW·L + passive area model)
- **Eye:** PRBS transient through channel model → eye height (V) and width (UI)

### Reward (first cut)
Per corner, a normalized signed margin per spec (positive = meets, capped). Aggregate
reward = weighted sum of the **worst corner's** margins, with a large bonus when *all*
specs pass at *all* corners. Details + rationale in `ROADMAP.md` Phase 3.

### Scope discipline
- Optimize the **CTLE first at TT** (Phase 1–2). Add DFE + PVT + noise/HD3 later.
- "Zero human intervention" is graded on the *loop*, not on us hand-tuning ranges. Keep
  ranges physically sane but let the agent search inside them.
