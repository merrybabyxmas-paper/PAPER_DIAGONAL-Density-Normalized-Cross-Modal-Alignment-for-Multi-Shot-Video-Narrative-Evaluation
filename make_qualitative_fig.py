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
        "model": "StoryDiffusion",
        "pattern": "Accumulation: A→AB→ABC",
        "entities": ["waiter", "manager", "child"],
        "S_ideal": [[1,0,0],[1,1,0],[1,1,1]],
        "S_obs":   [[1,0,0],[0,1,0],[1,1,1]],
        "shots": [KF/"storydiff_DM1K_422_Accumulation_124"/f"shot0{i}.png" for i in [1,2,3]],
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

def presence_str(row, ents):
    p = [ents[j] for j in range(len(row)) if row[j]]
    return ", ".join(p) if p else "∅"

fig = plt.figure(figsize=(8.5, 10.5))
gs = gridspec.GridSpec(4, 3, figure=fig, wspace=0.04, hspace=0.55,
                       left=0.01, right=0.99, top=0.97, bottom=0.02)

fig.patch.set_facecolor("white")

for ri, ex in enumerate(EXAMPLES):
    first_ax = None
    for ci in range(3):
        ax = fig.add_subplot(gs[ri, ci])
        img_path = ex["shots"][ci]
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            ax.imshow(img, aspect="auto")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        if ci == 0:
            first_ax = ax

        # Shot header (first row only)
        if ri == 0:
            ax.set_title(f"Shot {ci+1}", fontsize=10, fontweight="bold", pad=4)

        # S* / Ŝ below image
        ideal = presence_str(ex["S_ideal"][ci], ex["entities"])
        obs = presence_str(ex["S_obs"][ci], ex["entities"])
        match = (ex["S_ideal"][ci] == ex["S_obs"][ci])
        obs_color = "#C0392B" if not match else "#333"
        ax.text(0.5, -0.04, f"S*: {ideal}", transform=ax.transAxes,
                fontsize=6.5, ha="center", va="top", color="#333")
        ax.text(0.5, -0.15, f"Ŝ:  {obs}", transform=ax.transAxes,
                fontsize=6.5, ha="center", va="top", color=obs_color)

    # Row label above first image (reuse existing axis, don't create new one)
    first_ax.text(-0.02, 1.12, f"{ex['type']}  ({ex['model']}, {ex['pattern']})",
                  transform=first_ax.transAxes, fontsize=8, fontweight="bold",
                  va="bottom", ha="left", color="#2C3E50")

plt.savefig(str(OUT), dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved to {OUT}")
plt.close()
