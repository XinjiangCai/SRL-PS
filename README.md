# Stability-Guaranteed Reinforcement Learning for Power System Frequency Control
This repository implements a stability-guaranteed reinforcement learning algorithm for optimal primary frequency control in power systems with high penetration of inverter-based resources (IBRs).

## Key Features

- **Lyapunov-constrained neural controllers** ensuring system stability by construction
- **Smooth, monotonic control policies** with better approximation capacity than piecewise-linear controllers
- **Policy-gradient RL with RNN framework** embedding power system dynamics into RL training
- Benchmarked on **IEEE 39-bus system**, with comparisons to robust droop and PWL controllers

## Background

With the displacement of traditional synchronous generators by IBRs (e.g., renewable sources, batteries), conventional droop-based primary frequency control struggles to maintain stability. This work proposes a new approach combining reinforcement learning with stability certification via Lyapunov functions.

The controller is designed as a **smooth monotonic neural network**, and trained via a **physics-informed algorithm** that integrates swing equation dynamics directly into the recurrent training process.

## Getting Started

### Prerequisites

- Python 3.8+
- Recommended environment: `conda` or `venv`

### Installation

```bash
git clone https://github.com/XinjiangCai/Stable_RL_PWR_SYS.git

### Author
- Xinjiang Cai (xca@kth.se, truingcai@gmail.com)
- Former Research Engineer, Division of Electric Power and Energy Systems, EECS, KTH

