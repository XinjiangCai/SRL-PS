import numpy as np
import matplotlib.pyplot as plt
import os

# Load the total deviation data
droop_loss = np.load('droop_total_deviation.npy')
relu_loss = np.load('relu_total_deviation.npy')
softplus_loss = np.load('softplus_total_deviation.npy')

# Time line (assuming 0.01s timestep, and losses start from time step 1)
delta_t = 0.01
time_line = np.arange(1, len(droop_loss) + 1) * delta_t

# Set fonts and style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 14,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# Create output folder
os.makedirs("freqdev_plots", exist_ok=True)

# Plot
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

ax.plot(time_line, droop_loss*60, label="Robust linear droop", linestyle='-.', color='blue', linewidth=1.2)
ax.plot(time_line, relu_loss*60, label="Piecewise-linear", linestyle='--', color='green', linewidth=1.2)
ax.plot(time_line, softplus_loss*60, label="Smooth-nonlinear", linestyle='-', color='red', linewidth=1.2)

ax.set_xlabel('Time (s)')
ax.set_ylabel('TFD (Hz)')
# ax.set_title('Total Frequency Deviations Comparison')
ax.set_xlim([0, time_line[-1]])
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper center')

plt.tight_layout()
fig.savefig("frequency_deviation_comparison.png", dpi=300)
plt.close(fig)
