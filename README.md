# Activation Steering with a Feedback Controller

**[ICLR 2026]** | [Paper](https://openreview.net) | [Code](https://github.com/dungnvus/pid-steering)

> *We develop a control-theoretic foundation for activation steering, showing that popular methods correspond to proportional (P) controllers, and propose **PID Steering** — a principled framework leveraging the full PID controller for more robust and reliable LLM behavioral control.*

---

## Overview

Controlling the behavior of large language models (LLMs) is fundamental to safety alignment and reliable deployment. Existing steering methods lack theoretical performance guarantees and suffer from steady-state error — a well-known limitation of proportional (P) control.

We reframe activation steering as a **dynamical systems problem** and show that:

- **ActAdd**, **DirAblate**, and **Mean-AcT** are all instances of a P controller
- P controllers inherently admit a nonzero steady-state error due to system disturbances
- Adding **Integral (I)** action eliminates steady-state bias
- Adding **Derivative (D)** action reduces overshoot and improves stability

Our proposed **PID Steering** is lightweight, modular, and works as a drop-in replacement for the steering vector computation step in existing frameworks.

<p align="center">
  <img src="assets/overview.png" width="700"/>
</p>

---

## Method

Given an LLM with layers $f^{(k)}$, PID Steering constructs the steering vector at layer $k$ as:

$$u(k) = K_p r(k) + K_i \sum_{j=0}^{k-1} r(j) + K_d (r(k) - r(k-1))$$

where $r(k)$ is the difference-in-means error signal between contrastive datasets (e.g., harmful vs. harmless prompts), and $K_p, K_i, K_d \geq 0$ are the proportional, integral, and derivative gains.

- **P term**: Reacts immediately to the current error
- **I term**: Accumulates past errors to eliminate steady-state bias
- **D term**: Anticipates error trends to damp oscillations and reduce overshoot

---

## Repository Structure

This repo contains two subfolders, each covering a different set of experiments:

```
pid-steering/
├── llm-activation-control/    # Jailbreaking LLMs, built on Angular Steering
└── Mean-AcT/                  # Toxicity mitigation & diffusion style control, built on Mean-AcT
```

For setup and running instructions, please follow the **README in each subfolder**:

- **`llm-activation-control/`** — LLM jailbreaking experiments across Gemma2, LLaMA3, and Qwen2.5 (3B–14B)
- **`Mean-AcT/`** — LLM toxicity mitigation on RealToxicityPrompts, and image generation style control on FLUX.1-Schnell

## Citation

```bibtex
@inproceedings{nguyen2026pidsteering,
  title     = {Activation Steering with a Feedback Controller},
  author    = {Dung V. Nguyen and Nhi Y. Pham and Hieu M. Vu and Lei Zhang and Tan M. Nguyen},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```
