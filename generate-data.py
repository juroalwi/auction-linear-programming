import numpy as np
from model_solver import solve
from get_schools_grid import get_schools_grid
from get_competition_units import get_firms_with_competition_units
from make_plots import make_plots

bg_rgb = np.array([1.0, 1.0, 1.0])
schools_count = 709
grid_size = 1000
t = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(t, t)

f1_grid = Y >= np.clip(0.7 - 0.4 * X, 0.50, 0.65)
f2_grid = Y >= np.clip(0.0 + 1.8 * X, 0.2, 0.8)
f3_grid = Y >= np.clip(0.2 + 0.6 * X, 0.4, 1.0)
f4_grid = Y < np.clip(0.0 + 1.8 * X, 0.2, 0.8)
firms_grids = [f1_grid, f2_grid, f3_grid, f4_grid]
firms_count = len(firms_grids)

quantity_intervals = [
    [0, 50],
    [50, 100],
    [100, 200],
    [200, 300],
    [300, 500],
    [500, 709],
]
firms_price_per_interval = np.array(
    [
        [800, 700, 600, 200, 180, 160],
        [800, 800, 800, 800, 800, 300],
        [700, 550, 400, 280, 240, 220],
        [550, 450, 350, 300, 250, 240],
    ]
)

schools_grid = get_schools_grid(schools_count, grid_size)

(
    firms_map,
    cus_map,
    cus_img_grid,
) = get_firms_with_competition_units(firms_grids, schools_grid, grid_size, bg_rgb)

make_plots(firms_map, cus_map, cus_img_grid, schools_grid, bg_rgb)

solve(cus_map, firms_map, firms_price_per_interval, quantity_intervals, schools_count)
