#!/usr/bin/env python3
"""Generate qualitative failure figure for the paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

KF = Path("streamlit_app/data/keyframes")
OUT = Path("figures/qualitative_failures.png")

EXAMPLES = [
    {
        "type": "Type I: Context Bleeding",
        "model": "VGoT",
        "pattern": "Reduction: ABC→AB→A",
        "entities": ["cat", "dog", "parrot"],
        "S_ideal": [[1,1,1],[1,1,0],[1,0,0]],
        "S_obs":   [[0,1,1],[0,1,0],[0,1,0]],
        "shots": [KF/"vgot_DM1K_999_Reduction_009"/f"shot{i}.jpg" for i in [1,2,3]],
    },
    {
        "type": "Type II: Amnesia",
        "model": "StoryDiffusion",
        "pattern": "Convergence: A→B→AB",
        "entities": ["swimmer", "crab"],
        "S_ideal": [[1,0],[0,1],[1,1]],
        "S_obs":   [[1,0],[0,1],[0,1]],
        "shots": [KF/"storydiff_DM1K_726_Convergence_061"/f"shot0{i}.png" for i in [1,2,3]],
    },
    {
        "type": "Type III: Spurious",
        "model": "EchoShot",
        "pattern": "Convergence: A→B→AB",
        "entities": ["frisbee player", "spectator"],
        "S_ideal": [[1,0],[0,1],[1,1]],
        "S_obs":   [[1,0],[1,1],[1,0]],
        "shots": [KF/"echoshot_DM1K_989_Convergence_056"/f"shot{i}.jpg" for i in [1,2,3]],
    },
    {
        "type": "Type IV: Reversal",
        "model": "VIC",
        "pattern": "Split: AB→A→B",
        "entities": ["cook", "dishwasher"],
        "S_ideal": [[1,1],[1,0],[0,1]],
        "S_obs":   [[1,0],[1,1],[1,0]],
        "shots": [KF/"vic_DM1K_958_Split_069"/f"shot{i}.jpg" for i in [1,2,3]],
    },
]


fig = plt.figure(figsize=(8.5, 5.5))
gs = gridspec.GridSpec(4, 3, figure=fig, wspace=0.04, hspace=0.08,
                       left=0.04, right=0.99, top=0.94, bottom=0.02)

fig.patch.set_facecolor("white")

for ri, ex in enumerate(EXAMPLES):
    for ci in range(3):
        ax = fig.add_subplot(gs[ri, ci])
        img_path = ex["shots"][ci]
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            ax.imshow(img, aspect="auto")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        # Shot header (first row only)
        if ri == 0:
            ax.set_title(f"Shot {ci+1}", fontsize=10, fontweight="bold", pad=4)

        # Row label on left side (first column only)
        if ci == 0:
            ax.text(-0.05, 0.5, f"Row {ri+1}",
                    transform=ax.transAxes, fontsize=8, fontweight="bold",
                    va="center", ha="right", rotation=90, color="#2C3E50")

plt.savefig(str(OUT), dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved to {OUT}")
plt.close()
