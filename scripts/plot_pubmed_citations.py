"""
Plot cumulative PubMed citations per fiscal year

Source: https://www.nlm.nih.gov/bsd/medline_pubmed_production_stats.html
"""

import matplotlib.pyplot as plt
import seaborn as sns

YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
CUMULATIVE = [28_934_389, 30_178_674, 31_563_992, 33_136_289, 34_693_538, 36_555_430]

OUT_PATH: str = "pubmed_cumulative_citations.pdf"



sns.set_theme(context="paper", style="ticks", font="DejaVu Sans")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "font.size": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(5.2, 3.4))
color = "#2a6f97"

ax.plot(YEARS, [c / 1e6 for c in CUMULATIVE],
        marker="o", markersize=5.5, linewidth=1.6,
        color=color, markerfacecolor="white",
        markeredgecolor=color, markeredgewidth=1.4,
        clip_on=False, zorder=3
    )

for x, y in zip(YEARS, CUMULATIVE):
    ax.annotate(f"{y / 1e6:.1f}M",
                xy=(x, y / 1e6), xytext=(0, 8),
                textcoords="offset points",
                ha="center", fontsize=8.5, color="#333333"
            )

ax.set_xlabel("Fiscal year")
ax.set_ylabel("Cumulative PubMed citations (millions)")
ax.set_title("PubMed citation corpus growth, FY2018–FY2023", loc="left", pad=10)

ax.set_xticks(YEARS)
ax.margins(x=0.05)
ax.set_ylim(28, 38)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
ax.tick_params(direction="out", length=3)

fig.tight_layout()
fig.savefig(OUT_PATH)
fig.savefig(OUT_PATH.replace(".pdf", ".png"))
print(f"Saved: {OUT_PATH}")