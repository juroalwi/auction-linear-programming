## Internet service procurement — small testbed

Short, minimal sandbox to play with the model from:

- Bonomo, Catalán, Durán, Epstein, Guajardo, Jawtuschenko, Marenco. “An asymmetric multi‑item auction with quantity discounts applied to Internet service procurement in Buenos Aires public schools.” Annals of Operations Research. DOI: `https://doi.org/10.1007/s10479-016-2164-x`

The goal was to test the scenario of one firm having broad monopoly coverage on certain region. By running
the sandbox with the default parameters, one can see that firm 1 is awarded 33 schools in every possible solution,
even though it charges a high price for that quantity interval.

### What this does

- Builds a grid of schools and firms’ coverage regions.
- Creates competition units (CUs) with blended colors for visualization.
- Solves the allocation with quantity‑discount intervals using `PySCIPOpt`.
- Saves quick plots: `points.png` and `firms.png`.

### Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python index.py
```

Outputs will be in the project folder and the solver prints a short log.

Note: `PySCIPOpt` requires a working SCIP install. If you only want the visuals, you can comment the `solve(...)` line at the end of `generate-data.py`.

### Repo map

- `index.py`: builds the scenario, runs plots and solver.
- `get_schools_grid.py`: places schools on the grid.
- `get_competition_units.py`: computes CUs and colors.
- `make_plots.py`: renders `points.png` and `firms.png`.
- `model_solver.py`: SCIP model with quantity‑discount intervals.
