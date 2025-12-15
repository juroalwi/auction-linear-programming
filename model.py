from pyscipopt import Model
import numpy as np

model = Model("Licitación")

schools_count = 709
companies_count = 4
cu_count = 7
quantity_intervals_count = 3
company_price_per_interval = np.array(
    [
        [1000, 700, 600, 300],
        [800, 800, 800, 250],
        [700, 500, 400, 300],
        [600, 400, 300, 150],
    ]
)

x = np.empty(companies_count, cu_count)
y = np.empty(companies_count, quantity_intervals_count)
z = np.empty(companies_count, quantity_intervals_count)
for i in range(companies_count):
    for j in range(cu_count):
        x[i, j] = model.addVar(
            name=f"Número de escuelas asignadas a la empresa {i} en la unidad de competencia {j}",
            vartype="integer",
        )
for i in range(companies_count):
    for j in range(quantity_intervals_count):
        y[i, j] = model.addVar(
            name=f"A la empresa {i} se le asigna el intervalo {j}", vartype="binary"
        )
        z[i, j] = model.addVar(
            name=f"Número de escuelas asignadas a la empresa {i} en el intervalo {j}",
            vartype="integer",
        )

model.setObjective(
    model.sum(
        y[i, j] * company_price_per_interval[i, j]
        for i in range(companies_count)
        for j in range(quantity_intervals_count)
    ),
    "min",
)


model.optimize()
