import numpy as np
from pyscipopt import Model


def solve(cus_labels, cus_schools, cus_firms, firms_cus, M):
    model = Model("Licitación")
    model.setIntParam("display/verblevel", 0)

    cus_count = len(cus_labels)
    firms_count = len(firms_cus)
    quantity_intervals = [
        [0, 50],
        [50, 100],
        [100, 200],
        [200, 300],
        [300, 500],
        [500, 709],
    ]
    quantity_intervals_count = len(quantity_intervals)
    company_price_per_interval = np.array(
        [
            [800, 700, 600, 200, 180, 160],
            [800, 800, 800, 800, 800, 300],
            [700, 550, 400, 280, 240, 220],
            [550, 450, 350, 300, 250, 240],
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

    solutions = []
    prev_obj_val = np.inf
    iteration = 1
    max_iterations = 100

    while iteration <= max_iterations:
        print(f"Optimizing {iteration}...")
        model.optimize()
        status = model.getStatus()
        obj_val = model.getObjVal()

        if obj_val > prev_obj_val:
            print(f"Iteration {iteration} new worst objective value: {obj_val}")
            break

        prev_obj_val = obj_val

        if status != "optimal":
            print(f"Iteration {iteration} failed")
            break

        A = {}

        for i in range(firms_count):
            for j in firms_cus[i]:
                A[(i, j)] = round(model.getVal(x[i, j]))

        solutions.append(A)
        model.freeTransform()  # Return to problem stage before modifying

        W = np.empty((firms_count, cus_count), dtype=object)
        V = np.empty((firms_count, cus_count), dtype=object)

        for i in range(firms_count):
            for j in firms_cus[i]:
                if A[(i, j)] > 0:
                    print(
                        f"Empresa {i} en unidad de competencia {cus_labels[j]} tiene asignadas {A[(i, j)]} escuelas"
                    )
                    W[i, j] = model.addVar(name=f"W_{i}_{j}", vtype="B")
                    V[i, j] = model.addVar(name=f"V_{i}_{j}", vtype="I")
                    model.addCons(x[i, j] >= (A[(i, j)] + 1) * W[i, j])
                    model.addCons(M - x[i, j] >= (M - (A[(i, j)] - 1)) * V[i, j])

        model.addCons(
            sum(
                W[i, j] + V[i, j] if A[(i, j)] > 0 else 0
                for i in range(firms_count)
                for j in firms_cus[i]
            )
            >= 1
        )

        iteration += 1

    return True
