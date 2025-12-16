import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def make_plots(firms_map, cus_map, cus_img_grid, schools_grid, bg_rgb):
    grid_size = schools_grid.shape[0]
    plt.rc("font", family="serif")
    plt.figure(figsize=(6, 6))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Unidades de competencia")
    plt.tight_layout()
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="none",
            markerfacecolor="C0",
            markersize=5,
            label="Schools",
        )
    ]
    for cu in cus_map.values():
        handles.append(
            Patch(
                facecolor=cu["rgba"],
                edgecolor="black",
                label=cu["label"],
            )
        )
    plt.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0
    )

    # Competition units
    plt.gca().imshow(
        cus_img_grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        interpolation="nearest",
        zorder=2,
    )

    # Schools
    rows, cols = np.nonzero(schools_grid)
    schools_x = (cols + 0.5) / grid_size
    schools_y = (rows + 0.5) / grid_size
    plt.scatter(
        schools_x, schools_y, alpha=0.7, label="Escuelas", color="C0", s=10, zorder=3
    )
    plt.savefig("points.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Firms
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for ax, firm in zip(axes.flat, firms_map.values()):
        ax.set_title(firm["label"], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
        color_rgb = np.asarray(mcolors.to_rgb(firm["color"]), dtype=float)
        blended = color_rgb * 0.6 + bg_rgb * 0.4
        img = np.vstack([bg_rgb, blended])[firm["grid"].astype(int)]
        ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], interpolation="nearest")
    fig.subplots_adjust(
        left=0.15, right=0.85, top=0.70, bottom=0.08, wspace=0.25, hspace=0.30
    )
    plt.savefig("firms.png", dpi=200, bbox_inches="tight")
    plt.close()
