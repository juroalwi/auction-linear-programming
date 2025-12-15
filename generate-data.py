import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

bg_rgb = np.array([1.0, 1.0, 1.0])
schools_count = 12
grid_size = 5
t = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(t, t)

f1_grid = Y >= np.clip(0.7 - 0.4 * X, 0.50, 0.65)
f2_grid = Y >= np.clip(0.0 + 1.8 * X, 0.2, 0.8)
f3_grid = Y >= np.clip(0.2 + 0.6 * X, 0.4, 1.0)
f4_grid = Y < np.clip(0.0 + 1.8 * X, 0.2, 0.8)
firms_ids = ["C1", "C2", "C3", "C4"]
firms_grids = [f1_grid, f2_grid, f3_grid, f4_grid]
firms_colors = ["blue", "green", "yellow", "red"]
firms_cus = [[] for _ in range(len(firms_ids))]

# Generate random schools
rng = np.random.default_rng(42)
weights2 = np.exp(-0.25 * np.arange(grid_size))
weights1 = np.exp(0.25 * np.arange(grid_size))
p1 = weights1 / weights1.sum()
p2 = weights2 / weights2.sum()
schools_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
chosen = rng.choice(
    grid_size * grid_size, size=schools_count, replace=False, p=np.outer(p1, p2).ravel()
)
for school_id, flat_idx in enumerate(chosen, start=1):
    schools_grid.flat[flat_idx] = school_id

# Compute competition units
cus_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
for i, grid in enumerate(firms_grids):
    cus_grid |= grid.astype(np.uint8) << i
cus = np.unique(cus_grid)
cus = cus[cus > 0]
cus_labels = np.zeros((int(cus.max()) + 1), dtype=object)
cus_schools = np.zeros((int(cus.max()) + 1), dtype=object)
cus_firms = np.zeros((int(cus.max()) + 1), dtype=object)
cus_rgba = np.zeros((int(cus.max()) + 1, 3), dtype=float)
cus_rgba[0] = bg_rgb
for cu in cus:
    cu_grid = cus_grid == cu
    matching_schools_ids = schools_grid[cu_grid]
    matching_schools_ids = matching_schools_ids[matching_schools_ids > 0]
    matching_firms_ids = np.array([i for i in range(len(firms_ids)) if (cu >> i) & 1])
    cus_schools[cu] = matching_schools_ids
    cus_firms[cu] = matching_firms_ids + 1
    for firm_id in matching_firms_ids:
        firms_cus[firm_id].extend(matching_schools_ids)
    blended = bg_rgb.copy()
    for i in matching_firms_ids:
        color_rgb = np.asarray(mcolors.to_rgb(firms_colors[i]), dtype=float)
        blended = color_rgb * 0.25 + blended * 0.75
    cus_rgba[int(cu)] = blended
    cus_labels[int(cu)] = " + ".join(firms_ids[i] for i in matching_firms_ids)

############# Schools plot #############
plt.rc("font", family="serif")
plt.figure(figsize=(6, 6))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Schools locations")
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
for cu in cus:
    handles.append(
        Patch(
            facecolor=cus_rgba[cu],
            edgecolor="black",
            label=cus_labels[cu],
        )
    )
plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)

# Competition units
plt.gca().imshow(
    cus_rgba[cus_grid],
    origin="lower",
    extent=[0, 1, 0, 1],
    interpolation="nearest",
    zorder=2,
)

# Schools points
rows, cols = np.nonzero(schools_grid)
schools_x = (cols + 0.5) / grid_size
schools_y = (rows + 0.5) / grid_size
plt.scatter(
    schools_x, schools_y, alpha=0.7, label="Schools", color="C0", s=10, zorder=3
)
plt.savefig("points.png", dpi=200, bbox_inches="tight")
plt.close()

############# Firms plot #############
fig, axes = plt.subplots(1, 4, figsize=(8, 2))
for ax, name, color, grid in zip(axes, firms_ids, firms_colors, firms_grids):
    ax.set_title(name, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)
    color_rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    blended = color_rgb * 0.6 + bg_rgb * 0.4
    img = np.vstack([bg_rgb, blended])[grid.astype(int)]
    ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], interpolation="nearest")
plt.tight_layout()
plt.savefig("firms.png", dpi=200, bbox_inches="tight")
plt.close()
