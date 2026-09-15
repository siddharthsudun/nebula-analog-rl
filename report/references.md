# silQ references

IEEE style. Groups [1]–[19] are what the project actually uses: the brief, the circuit background, the process, the simulator, the methods and the libraries. Group [20]–[33] is related work we compare against. It gives context and doesn't back any silQ number.

On 13 Sept 2026 I checked the related-work papers, Stable-Baselines3, Gymnasium and Razavi's DFE article against their arXiv abstract pages, the JMLR page or a publisher listing. The textbook, the PPO, CMA-ES, TPE, Optuna, PyTorch, NumPy and SciPy entries are standard citations written from memory, and the PCIe year wasn't checked either. Check those before printing. Software versions are the exact pins in `requirements.txt` and `requirements-dashboard.txt`. Two entries have no author list because their publisher pages wouldn't load; they're marked below.

A matching BibTeX file is [references.bib](references.bib).

## Problem statement and standards

[1] Astera Labs, "AI/ML for Analog Circuit Design," problem statement, Analog Track, Nebula 2026, BITS Pilani K K Birla Goa Campus, 2026. (Transcribed in `docs/PROBLEM.md`.)

[2] PCI-SIG, *PCI Express Base Specification, Revision 2.0*, 2007.

## Equalizer circuit background

[3] B. Razavi, *Design of Integrated Circuits for Optical Communications*, 2nd ed. Hoboken, NJ, USA: Wiley, 2012.

[4] B. Razavi, "The decision-feedback equalizer [A Circuit for All Seasons]," *IEEE Solid-State Circuits Magazine*, vol. 9, no. 4, pp. 13–16, Fall 2017, doi: 10.1109/MSSC.2017.2745939.

## Process and simulation

[5] SkyWater Technology Foundry and Google, "SkyWater SKY130 open source process design kit." [Online]. Available: https://github.com/google/skywater-pdk

[6] The ngspice developers, *Ngspice User's Manual*. [Online]. Available: https://ngspice.sourceforge.io/docs.html (libngspice 41 used for the recorded results on Windows; ngspice 47 on macOS.)

[7] F. Salvaire, "PySpice: Simulate electronic circuits using Python and the Ngspice/Xyce simulators," version 1.5. [Online]. Available: https://github.com/PySpice-org/PySpice

## Learning and optimization methods

[8] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.

[9] N. Hansen and A. Ostermeier, "Completely derandomized self-adaptation in evolution strategies," *Evolutionary Computation*, vol. 9, no. 2, pp. 159–195, 2001.

[10] N. Hansen, "The CMA evolution strategy: A tutorial," arXiv:1604.00772, 2016.

[11] J. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, "Algorithms for hyper-parameter optimization," in *Advances in Neural Information Processing Systems 24 (NeurIPS)*, 2011.

[12] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining*, 2019, pp. 2623–2631.

## Software

[13] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable reinforcement learning implementations," *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021. (Version 2.9.0 used.)

[14] M. Towers *et al.*, "Gymnasium: A standard interface for reinforcement learning environments," arXiv:2407.17032, 2024. (Version 1.3.0 used.)

[15] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019. (Version 2.13.0 used.)

[16] N. Hansen, Y. Akimoto, and P. Baudis, "CMA-ES/pycma." [Online]. Available: https://github.com/CMA-ES/pycma (Version 4.4.4 used.)

[17] C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.

[18] P. Virtanen *et al.*, "SciPy 1.0: Fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, pp. 261–272, 2020.

[19] Anthropic, "Claude API documentation: Messages API." [Online]. Available: https://docs.anthropic.com (Used by the optional "Read with Claude" brief reader.)

## Related work: learning-based analog sizing

[20] K. Settaluri, A. Haj-Ali, Q. Huang, K. Hakhamaneshi, and B. Nikolić, "AutoCkt: Deep reinforcement learning of analog circuit designs," in *Proc. Design, Automation & Test in Europe (DATE)*, 2020. arXiv:2001.01808.

[21] H. Wang, K. Wang, J. Yang, L. Shen, N. Sun, H.-S. Lee, and S. Han, "GCN-RL circuit designer: Transferable transistor sizing with graph neural networks and reinforcement learning," in *Proc. 57th ACM/IEEE Design Automation Conf. (DAC)*, 2020. arXiv:2005.00406.

[22] K. Hakhamaneshi, N. Werblun, P. Abbeel, and V. Stojanović, "BagNet: Berkeley analog generator with layout optimizer boosted with deep neural networks," in *Proc. IEEE/ACM Int. Conf. Computer-Aided Design (ICCAD)*, 2019. arXiv:1907.10515.

[23] S. Zhang, W. Lyu, F. Yang, C. Yan, D. Zhou, X. Zeng, and X. Hu, "An efficient multi-fidelity Bayesian optimization approach for analog circuit synthesis," in *Proc. 56th ACM/IEEE Design Automation Conf. (DAC)*, 2019. arXiv:1912.00392.

[24] A. F. Budak, P. Bhansali, B. Liu, N. Sun, D. Z. Pan, and C. V. Kashyap, "DNN-Opt: An RL inspired optimization for analog circuit sizing using deep neural networks," in *Proc. 58th ACM/IEEE Design Automation Conf. (DAC)*, 2021. arXiv:2110.00211.

[25] W. Cao, J. Gao, T. Ma, R. Ma, M. Benosman, and X. Zhang, "RoSE-Opt: Robust and efficient analog circuit parameter optimization with knowledge-infused reinforcement learning," *IEEE Trans. Computer-Aided Design of Integrated Circuits and Systems*. arXiv:2407.19150.

## Related work: PVT-aware sizing

[26] Kong *et al.*, "PVTSizing: A TuRBO-RL-based batch-sampling optimization framework for PVT-robust analog circuit synthesis," in *Proc. 61st ACM/IEEE Design Automation Conf. (DAC)*, 2024, doi: 10.1145/3649329.3661850. *Full author list not retrieved: the ACM page refused the request.*

[27] D. Kim, J. Park, C. Shin, J. Jung, K. Shin, S. Baek, S. Heo, W. Kim, I. Jeong, J. Cho, and J. Park, "GLOVA: Global and local variation-aware analog circuit design with risk-sensitive reinforcement learning," in *Proc. 62nd ACM/IEEE Design Automation Conf. (DAC)*, 2025. arXiv:2505.11208.

[28] S. Kim, Z. Wang, S. Lee, Y. Oh, H. Zhu, D. Kim, and D. Z. Pan, "PPAAS: PVT and Pareto aware analog sizing via goal-conditioned reinforcement learning," in *Proc. 44th IEEE/ACM Int. Conf. Computer-Aided Design (ICCAD)*, 2025. arXiv:2507.17003.

[29] "HR2: A PVT aware hierarchical RL based sizing framework for robust analog circuit design," in *Proc. 7th ACM/IEEE Symp. Machine Learning for CAD (MLCAD)*, 2025, doi: 10.1109/MLCAD65511.2025.11189231. *Title, venue and DOI from a search listing; authors not retrieved because the IEEE Xplore page didn't load.*

## Related work: LLM-assisted sizing

[30] D. V. Kochar, H. Wang, A. Chandrakasan, and X. Zhang, "LEDRO: LLM-enhanced design space reduction and optimization for analog circuits," in *Proc. IEEE Int. Conf. LLM-Aided Design (ICLAD)*, 2025. arXiv:2411.12930.

[31] M. Ahmadzadeh, K. Chen, and G. Gielen, "AnaFlow: Agentic LLM-based workflow for reasoning-driven explainable and sample-efficient analog circuit sizing," in *Proc. IEEE/ACM Int. Conf. Computer-Aided Design (ICCAD)*, 2025. arXiv:2511.03697.

[32] J. Chen, S. Li, and X. He, "White-box reasoning: Synergizing LLM strategy and gm/Id data for automated analog circuit design," arXiv:2508.13172, 2025.

[33] A. J. Bujana and A. I. Karsilayan, "A self-calibrating framework for analog circuit sizing using LLM-derived analytical equations," arXiv:2604.07387, 2026.

## Left out on purpose

- **FALCON** (NeurIPS 2025, arXiv:2505.21923, checked), **AutoCircuit-RL** (arXiv:2506.03122) and **KATO** (arXiv:2404.14433), the last two not checked. These cover layout, topology generation and transfer across process nodes. silQ doesn't do any of those, so citing them would suggest overlap that isn't there.
- **Lighthouse RL** (arXiv:2607.14008) and **SABLE** (arXiv:2607.03701). Both are July 2026 preprints with no venue yet. Add them if the report needs to show the newest work.
- **cVAE design portfolios** (arXiv:2510.05160). This is only a future-work idea in `docs/STATUS.md`.
- **FD-MAGRPO** (AAAI) and **AutoSizer**. `docs/NOVELTY.md` gives no usable identifier for either, and I couldn't confirm them.
