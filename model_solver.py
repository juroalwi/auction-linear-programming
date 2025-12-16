import numpy as np
from pyscipopt import Model


def solve(cus_map, firms_map, firms_price_per_interval, quantity_intervals, M):
    model = Model("Licitación")
    model.setIntParam("display/verblevel", 0)

    cus_count = len(cus_map)
    firms_count = len(firms_map)
    quantity_intervals_count = len(quantity_intervals)

    x = np.empty((firms_count, cus_count), dtype=object)
    y = np.empty((firms_count, quantity_intervals_count), dtype=object)
    z = np.empty((firms_count, quantity_intervals_count), dtype=object)

    for i, firm in firms_map.items():
        for j in firm["cus"]:
            x[i, j] = model.addVar(
                name=f"Número de escuelas asignadas a la empresa {firm['label']} en la unidad de competencia {cus_map[j]['label']}",
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
            z[i, t] * firms_price_per_interval[i, t]
            for i in range(firms_count)
            for t in range(quantity_intervals_count)
        ),
        "minimize",
    )

    for j, cu in cus_map.items():
        model.addCons(sum(x[i, j] for i in cu["firms"]) == len(cu["schools"]))

    for i, firm in firms_map.items():
        for t in range(quantity_intervals_count):
            model.addCons(
                sum(x[i, j] for j in firm["cus"])
                >= quantity_intervals[t][0] - M * (1 - y[i, t])
            )
            model.addCons(
                sum(x[i, j] for j in firm["cus"])
                <= quantity_intervals[t][1] + M * (1 - y[i, t])
            )
            model.addCons(
                sum(x[i, j] for j in firm["cus"]) <= z[i, t] + M * (1 - y[i, t])
            )

    for i in range(firms_count):
        model.addCons(sum(y[i, t] for t in range(quantity_intervals_count)) == 1)

    solutions = []
    prev_obj_val = np.inf
    iteration = 1
    max_iterations = 100
    eps = 1e-6

    while iteration <= max_iterations:
        print(f"Optimizando {iteration}...")
        model.optimize()
        status = model.getStatus()
        obj_val = model.getObjVal()

        if obj_val > prev_obj_val + eps:
            print(
                f"Nuevo valor objetivo {obj_val} es peor que el anterior {prev_obj_val}. Abortando."
            )
            break

        prev_obj_val = obj_val

        if status != "optimal":
            print(f"Iteration {iteration} failed")
            break

        A = {}

        for i, firm in firms_map.items():
            for j in firm["cus"]:
                A[(i, j)] = round(model.getVal(x[i, j]))

        solutions.append(A)
        model.freeTransform()  # Return to problem stage before modifying

        W = np.empty((firms_count, cus_count), dtype=object)
        V = np.empty((firms_count, cus_count), dtype=object)

        for i, firm in firms_map.items():
            for j in firm["cus"]:
                if A[(i, j)] > 0:
                    print(
                        f"Empresa {firm['label']} en unidad de competencia {cus_map[j]['label']} tiene asignadas {A[(i, j)]} escuelas"
                    )
                    W[i, j] = model.addVar(name=f"W_{i}_{j}", vtype="B")
                    V[i, j] = model.addVar(name=f"V_{i}_{j}", vtype="I")
                    model.addCons(x[i, j] >= (A[(i, j)] + 1) * W[i, j])
                    model.addCons(M - x[i, j] >= (M - (A[(i, j)] - 1)) * V[i, j])

        model.addCons(
            sum(
                W[i, j] + V[i, j] if A[(i, j)] > 0 else 0
                for i, firm in firms_map.items()
                for j in firm["cus"]
            )
            >= 1
        )

        iteration += 1

    return True
