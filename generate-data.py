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
r1_grid = Y >= np.clip(0.7 - 0.4 * X, 0.50, 0.65)
r2_grid = Y >= np.clip(0.0 + 1.8 * X, 0.2, 0.8)
r3_grid = Y >= np.clip(0.2 + 0.6 * X, 0.4, 1.0)
r4_grid = Y < np.clip(0.0 + 1.8 * X, 0.2, 0.8)
regions_grids = [r1_grid, r2_grid, r3_grid, r4_grid]
regions_colors = ["blue", "green", "yellow", "red"]
regions_names = ["C1", "C2", "C3", "C4"]
regions_count = len(regions_grids)
regions_schools = np.empty(regions_count, dtype=object)

regions = {
    "C1": {
        "color": "blue",
        "name": "C1",
        "grid": r1_grid,
        "schools": [],
    },
    "C2": {
        "color": "green",
        "name": "C2",
        "grid": r2_grid,
        "schools": [],
    },
    "C3": {
        "color": "yellow",
        "name": "C3",
        "grid": r3_grid,
        "schools": [],
    },
    "C4": {
        "color": "red",
        "name": "C4",
        "grid": r4_grid,
        "schools": [],
    },
}

# Generate random schools
rng = np.random.default_rng(42)
weights2 = np.exp(-0.25 * np.arange(grid_size))
weights1 = np.exp(0.25 * np.arange(grid_size))
p1 = weights1 / weights1.sum()
p2 = weights2 / weights2.sum()
schools_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
w_flat = np.outer(p1, p2).ravel()
chosen = rng.choice(grid_size * grid_size, size=schools_count, replace=False, p=w_flat)
for school_id, flat_idx in enumerate(chosen, start=1):
    schools_grid.flat[flat_idx] = school_id

# Generate intersections
intersections_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
for i, grid in enumerate(regions_grids):
    intersections_grid |= grid.astype(np.uint8) << i
intersections = np.unique(intersections_grid)
intersections = intersections[intersections > 0]
intersections_labels = np.empty((int(intersections.max()) + 1), dtype=object)
intersections_rgba = np.zeros((int(intersections.max()) + 1, 3), dtype=float)
intersections_rgba[0] = bg_rgb
for intersection in intersections:
    matching_regions_ids = [i for i in range(regions_count) if (intersection >> i) & 1]
    blended = bg_rgb.copy()
    for i in matching_regions_ids:
        color_rgb = np.asarray(mcolors.to_rgb(regions_colors[i]), dtype=float)
        blended = color_rgb * 0.25 + blended * 0.75
    intersections_rgba[int(intersection)] = blended
    intersections_labels[int(intersection)] = " + ".join(
        regions_names[i] for i in matching_regions_ids
    )


############# Schools coordinates #############
# np.savetxt(
#     "schools.csv",
#     np.column_stack((x, y)),
#     delimiter=",",
#     header="x,y",
#     comments="",
#     fmt="%.6f",
# )
# print("Saved schools coordinates to schools.csv")

for intersection in intersections:
    intersection_grid = intersections_grid == intersection
    matching_schools_ids = schools_grid[intersection_grid]
    matching_schools_ids = matching_schools_ids[matching_schools_ids > 0]
    matching_regions_ids = [i for i in range(regions_count) if (intersection >> i) & 1]
    for region_id in matching_regions_ids:
        regions[regions_names[region_id]]["schools"].extend(matching_schools_ids)

print(regions)

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
for intersection in intersections:
    handles.append(
        Patch(
            facecolor=intersections_rgba[intersection],
            edgecolor="black",
            label=intersections_labels[intersection],
        )
    )
plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)

# Competition units
plt.gca().imshow(
    intersections_rgba[intersections_grid],
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

############# Regions plot #############
fig, axes = plt.subplots(1, 4, figsize=(8, 2))
for ax, name, color, grid in zip(axes, regions_names, regions_colors, regions_grids):
    ax.set_title(name, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)
    # Render single-region mask
    color_rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    blended = color_rgb * 0.6 + bg_rgb * 0.4
    img = np.vstack([bg_rgb, blended])[grid.astype(int)]
    ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], interpolation="nearest")
plt.tight_layout()
plt.savefig("regions.png", dpi=200, bbox_inches="tight")
plt.close()
