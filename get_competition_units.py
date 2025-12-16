import numpy as np
import matplotlib.colors as mcolors

colors_saturation = 0.65
colors_value = 0.9


def get_firms_with_competition_units(
    firms_grids, schools_grid, grid_size, default_bg_rgb
):
    firms_count = len(firms_grids)
    hues = np.linspace(0.0, 1.0, firms_count, endpoint=False)
    firms_colors = [
        tuple(mcolors.hsv_to_rgb((h, colors_saturation, colors_value))) for h in hues
    ]
    firms_labels = [f"Firma {i}" for i in range(firms_count)]
    firms_cus = [[] for _ in range(firms_count)]

    cus_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for i, grid in enumerate(firms_grids):
        cus_grid |= grid.astype(np.uint8) << i
    cus = np.unique(cus_grid)
    cus = cus[cus > 0]
    cus_labels = np.zeros((int(cus.max()) + 1), dtype=object)
    cus_schools = np.zeros((int(cus.max()) + 1), dtype=object)
    cus_firms = np.zeros((int(cus.max()) + 1), dtype=object)
    cus_rgba = np.zeros((int(cus.max()) + 1, 3), dtype=float)
    cus_rgba[0] = default_bg_rgb
    for cu_idx, cu in enumerate(cus):
        cu_grid = cus_grid == cu
        matching_schools_ids = schools_grid[cu_grid]
        matching_schools_ids = matching_schools_ids[matching_schools_ids > 0]
        matching_firms_ids = np.array([i for i in range(firms_count) if (cu >> i) & 1])
        cus_schools[cu] = matching_schools_ids
        cus_firms[cu] = matching_firms_ids
        for idx in matching_firms_ids:
            firms_cus[idx].append(cu_idx)
        blended = default_bg_rgb.copy()
        for i in matching_firms_ids:
            color_rgb = np.asarray(mcolors.to_rgb(firms_colors[i]), dtype=float)
            blended = color_rgb * 0.25 + blended * 0.75
        cus_rgba[int(cu)] = blended
        cus_labels[int(cu)] = " + ".join(f"C{i}" for i in matching_firms_ids)

    cus_map = {}
    for idx, (label, schools, firms, rgba) in enumerate(
        zip(cus_labels[cus], cus_schools[cus], cus_firms[cus], cus_rgba[cus])
    ):
        cus_map[idx] = {
            "label": label,
            "schools": schools,
            "firms": firms,
            "rgba": rgba,
        }

    firms_map = {}
    for idx, (grid, label, cus, color) in enumerate(
        zip(firms_grids, firms_labels, firms_cus, firms_colors)
    ):
        firms_map[idx] = {
            "grid": grid,
            "label": label,
            "cus": cus,
            "color": color,
        }
    return (firms_map, cus_map, cus_rgba[cus_grid])
