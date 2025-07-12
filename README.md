# Stability-Guaranteed Reinforcement Learning for Power System Frequency Control
This repository implements a stability-guaranteed reinforcement learning algorithm for optimal primary frequency control in power systems with high penetration of inverter-based resources (IBRs).

## Key Features

- **Lyapunov-constrained neural controllers** ensuring system stability by construction
- **Smooth, monotonic control policies** with better approximation capacity than piecewise-linear controllers
- **Policy-gradient RL with RNN framework** embedding power system dynamics into RL training
- Benchmarked on **IEEE 39-bus system**, with comparisons to robust droop and PWL controllers

## Background

With the displacement of traditional synchronous generators by IBRs (e.g., renewable sources, batteries), flexibly regulating active power supply of these resources can help maintain system stability. This work proposes a new approach combining reinforcement learning with stability certification via Lyapunov functions.

The controller is designed as a **smooth monotonic neural network**, and trained via a **RNN-based structure** that integrates swing equation dynamics directly into the recurrent training process to capture time-dependent patterns for effective learning.

## Project Structure
SRL-PS/
│
├── data/                            # Power system data (e.g., IEEE 39-bus Kron-reduced)
│   └── IEEE_39bus_Kron.mat
│
├── env/                             # Common simulation environment and dynamics
│   └── interact_env.py
│
├── robust_droop/                    # Robust linear droop controller implementation
│   ├── main.py                      # Evaluation script
│   └── train.py                     # RL-style training with droop controller
│
├── piecewise_linear/               # Piecewise-linear stabilizing controller (baseline)
│   ├── main.py
│   └── train.py
│
├── smooth_nonlinear/               # Proposed smooth-nonlinear monotonic controller
│   ├── main.py
│   └── train.py
│
├── vis/                             # Visualization of training results and comparisons
│   ├── freqdev_comp/                # Frequency deviation comparison
│   │   ├── *.npy                    # Raw TFD data
│   │   ├── *.png                    # Plots of deviation over time
│   │   └── vis.py                   # Plotting script for TFD
│   │
│   └── loss_comp/                   # Training loss comparison across methods
│       ├── *.npy
│       ├── *.png
│       └── loss_vis.py
│
├── LICENSE                          # MIT or other license file
└── README.md                        # Project introduction and usage guide

## Getting Started

### Prerequisites

- Python 3.8+
- Pytorch 2.5.*
- Recommended environment: `conda`
  

### Installation

```bash
git clone https://github.com/XinjiangCai/Stable_RL_PWR_SYS.git
```

## Author
- Xinjiang Cai (xca@kth.se, truingcai@gmail.com)
- Former Research Engineer, Division of Electric Power and Energy Systems, EECS, KTH

## Acknowledgement
The work was finished independently by the author during a stay at KTH Royal Institute of Technology.
The data were from open-source resources, and the code was inspired by https://github.com/Wenqi-Cui/RNN-RL-Frequency-Lyapunov.
The computational resources were provided by Hangzhou Innovation Institute of Beihang University.
