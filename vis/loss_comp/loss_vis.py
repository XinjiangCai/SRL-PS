import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as patches

# Load loss records
droop_loss = np.load('droop_loss_record.npy')
relu_loss = np.load('relu_loss_record.npy')
softplus_loss = np.load('softplus_loss_record.npy')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

# Create figure
fig, ax = plt.subplots(figsize=(5, 4))

# Main loss curves
ax.plot(droop_loss, label='Robust linear droop controller', linestyle='-', color='tab:blue', linewidth=2)
ax.plot(relu_loss, label='Piecewise-linear controller', linestyle='--', color='tab:orange', linewidth=2)
ax.plot(softplus_loss, label='Smooth-nonlinear controller', linestyle='-.', color='tab:green', linewidth=2)

# Labels
ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
# ax.set_title('Training Loss Comparison', fontsize=15)
ax.set_xlim(0, 1000)
ax.grid(True)
ax.legend(fontsize=10)

# === Create zoomed-in inset ===
axins = inset_axes(ax, width="65%", height="65%", loc='lower left', 
                   bbox_to_anchor=(0.2, 0.2, 1.1, 0.8),  # [left, bottom, width, height]
                   bbox_transform=ax.transAxes)

# Plot inside zoom
episodes_zoom = np.arange(700, 1000)
axins.plot(episodes_zoom, droop_loss[700:1000], linestyle='-', color='tab:blue', linewidth=1)
axins.plot(episodes_zoom, relu_loss[700:1000], linestyle='-', color='tab:orange', linewidth=1)
axins.plot(episodes_zoom, softplus_loss[700:1000], linestyle='-', color='tab:green', linewidth=1)

axins.set_xlim(700, 1000)
axins.set_ylim(
    min(min(droop_loss[700:1000]), min(relu_loss[700:1000]), min(softplus_loss[700:1000])) * 0.8,
    max(max(droop_loss[700:1000]), max(relu_loss[700:1000]), max(softplus_loss[700:1000])) * 1.2
)
axins.grid(True)

# Connect upper-left (x1, y1) in original -> lower-left (x2, y2) in zoom
xy1 = (700, 0.4)  # original plot (episode=700, loss ~0.3)
xy2 = (700, 0.145)  # zoomed plot (same data point)

con1 = ConnectionPatch(xyA=xy2, coordsA=axins.transData,
                       xyB=xy1, coordsB=ax.transData,
                       color="gray", linewidth=0.8)
fig.add_artist(con1)

# Connect upper-right (x1, y1) in original -> lower-right (x2, y2) in zoom
xy1 = (1000, 0.4)  # original plot (episode=1000)
xy2 = (1000, 0.145)  # zoomed plot

con2 = ConnectionPatch(xyA=xy2, coordsA=axins.transData,
                       xyB=xy1, coordsB=ax.transData,
                       color="gray", linewidth=0.8)
fig.add_artist(con2)

# Add a rectangle manually to the main plot
rect = patches.Rectangle((700, 0.4),  # (x,y) lower-left corner
                          300,          # width (1000-700)
                          -0.255,  # height
                          linewidth=1,
                          edgecolor='black',
                          facecolor='none',
                          linestyle='-')
ax.add_patch(rect)

# Mark the zoomed area with lines
# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=1, linestyle='none')

plt.tight_layout()
plt.savefig('Loss_Comparison_with_Zoom.png', dpi=300, bbox_inches='tight')

# plt.show()