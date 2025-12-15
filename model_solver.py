import numpy as np
from pyscipopt import Model


def solve(cus_labels, cus_schools, cus_firms, firms_cus, M):
    cus_count = len(cus_labels)
    firms_count = len(firms_cus)

    model = Model("Licitación")
    quantity_intervals = [[0, 100], [100, 200], [200, 400], [400, 709]]
    quantity_intervals_count = len(quantity_intervals)
    company_price_per_interval = np.array(
        [
            [10, 10, 10, 10],
            [800, 800, 800, 250],
            [700, 500, 400, 300],
            [10, 10, 10, 10],
        ]
    )

    x = np.empty((firms_count, cus_count), dtype=object)
    y = np.empty((firms_count, quantity_intervals_count), dtype=object)
    z = np.empty((firms_count, quantity_intervals_count), dtype=object)

    for i in range(firms_count):
        for j in firms_cus[i]:
            x[i, j] = model.addVar(
                name=f"Número de escuelas asignadas a la empresa {i} en la unidad de competencia {cus_labels[j]}",
                vtype="I",
            )

    for i in range(firms_count):
        for t in range(quantity_intervals_count):
            y[i, t] = model.addVar(
                name=f"A la empresa {i} se le asigna el intervalo {quantity_intervals[t]}",
                vtype="B",
            )
            z[i, t] = model.addVar(
                name=f"Número de escuelas asignadas a la empresa {i} en el intervalo {quantity_intervals[t]}",
                vtype="I",
            )

    model.setObjective(
        sum(
            z[i, t] * company_price_per_interval[i, t]
            for i in range(firms_count)
            for t in range(quantity_intervals_count)
        ),
        "minimize",
    )

    for j in range(cus_count):
        model.addCons(sum(x[i, j] for i in cus_firms[j]) == len(cus_schools[j]))

    for i in range(firms_count):
        for t in range(quantity_intervals_count):
            model.addCons(
                sum(x[i, j] for j in firms_cus[i])
                >= quantity_intervals[t][0] - M * (1 - y[i, t])
            )
            model.addCons(
                sum(x[i, j] for j in firms_cus[i])
                <= quantity_intervals[t][1] + M * (1 - y[i, t])
            )
            model.addCons(
                sum(x[i, j] for j in firms_cus[i]) <= z[i, t] + M * (1 - y[i, t])
            )

    for i in range(firms_count):
        model.addCons(sum(y[i, t] for t in range(quantity_intervals_count)) == 1)

    model.optimize()
    solutions = []

    while model.getNSols() > 0:
        if model.getNSols() == 0:
            break
        sol = model.getBestSol()
        print(f"Solution: {sol}")
        solutions.append(sol)
        model.freeTransform()  # Return to problem stage before modifying

        W = np.empty((firms_count, cus_count), dtype=object)
        V = np.empty((firms_count, cus_count), dtype=object)

        for i in range(firms_count):
            for j in firms_cus[i]:
                W[i, j] = model.addVar(name=f"W_{i}_{j}", vtype="B")
                V[i, j] = model.addVar(name=f"V_{i}_{j}", vtype="I")

        for i in range(firms_count):
            for j in firms_cus[i]:
                if sol[x[i, j]] > 0:
                    model.addCons(x[i, j] >= (sol[x[i, j]] + 1) * W[i, j])
                    model.addCons(M - x[i, j] >= (M - (sol[x[i, j]] - 1)) * V[i, j])

        model.addCons(
            sum(
                W[i, j] + V[i, j] if sol[x[i, j]] > 0 else 0
                for i in range(firms_count)
                for j in firms_cus[i]
            )
            >= 1
        )

        model.optimize()

    return True
