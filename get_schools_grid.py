import numpy as np

rng = np.random.default_rng(42)


def clip(u):
    return np.clip(u, 1e-9, 1 - 1e-9)


def beta_shape(u, a, b):
    return clip(u) ** (a - 1.0) * ((1.0 - clip(u)) ** (b - 1.0))


def gaussian(u, mu, sigma):
    return np.exp(-0.5 * ((u - mu) / sigma) ** 2)


def get_schools_grid(schools_count, grid_size):
    centers = (np.arange(grid_size) + 0.5) / grid_size
    w1 = 0.8 * beta_shape(centers, 1.0, 4.0) + 0.2 * gaussian(centers, 0.0, 1)
    w2 = 0.6 * beta_shape(centers, 4.0, 1) + 0.4 * gaussian(centers, 2.0, 1)
    p1 = w1[::-1] / w1.sum()
    p2 = w2[::-1] / w2.sum()

    schools_grid = np.zeros((grid_size, grid_size), dtype=np.int32)

    chosen = rng.choice(
        grid_size * grid_size,
        size=schools_count,
        replace=False,
        p=np.outer(p1, p2).ravel(),
    )

    for school_id, flat_idx in enumerate(chosen, start=1):
        schools_grid.flat[flat_idx] = school_id

    return schools_grid
