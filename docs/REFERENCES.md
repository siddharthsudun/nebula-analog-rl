# References

## Core — read these first
- **AutoCkt: Deep Reinforcement Learning of Analog Circuit Designs** — Settaluri,
  Hakhamaneshi, Han, Nikolić (DATE 2020). Sparse-subsampling RL agent that sizes analog
  circuits via SPICE in an OpenAI Gym loop. The closest published analogue to this
  competition. Has open-source code — study its env/reward design.
- **GCN-RL Circuit Designer** — Wang, Han et al. (DAC 2020). Graph conv net + RL,
  transferable across topologies and technology nodes.
- **DNN-Opt / BagNet** — sample-efficient analog sizing; useful for the "fewer sims"
  argument.

## Equalizer / CTLE background
- Razavi, *Design of Integrated Circuits for Optical Communications* — CTLE and source
  degeneration fundamentals.
- Any SerDes RX front-end reference on CTLE + DFE (search: "CTLE source degeneration
  peaking zero", "1-tap DFE first post-cursor").
- PCIe Gen 2 = 5.0 GT/s, Nyquist = 2.5 GHz. NRZ.

## Tools
- **ngspice** — open SPICE. `brew install ngspice`.
- **PySpice** — Python wrapper over ngspice/Xyce (optional; raw ngspice is fine too).
- **SKY130 PDK** — open_pdks / volare; SPICE models for the 130 nm process.
- **IHP sg13g2** — open 130 nm BiCMOS PDK (alternative; has real BJTs).
- **Gymnasium** — successor to OpenAI Gym; env API.
- **stable-baselines3** — PPO, DDPG, SAC implementations.
- **Optuna / scikit-optimize** — Bayesian-optimization baseline for the comparison.

## Anthropic API (for the LLM wrapper bonus)
- Use the Claude Messages API; model id `claude-opus-4-8` (or a smaller model for the
  parser). Load the `claude-api` skill before writing that code.

## Notes
- Search terms that pull good prior art: "reinforcement learning analog sizing",
  "AutoCkt", "transistor sizing deep RL", "analog design automation Gym ngspice".
