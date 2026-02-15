import numpy as np
import pyomo.environ as pyo

def product_mix(model):
    solver = pyo.SolverFactory("glpk")

    # variables
    model.products = pyo.Set(initialize=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    model.p_quantity = pyo.Var(model.products, domain=pyo.NonNegativeReals)

    # parameters
    prices_data = {
        'A': 100, 'B': 120, 'C': 135, 'D': 90, 'E': 125, 'F': 110, 'G': 105
    }

    model.prices = pyo.Param(model.products, initialize=prices_data)

    model.materials = pyo.Set(initialize=['M1', 'M2', 'M3'])
    required_material = {
        ('A', 'M1'): 0, ('A', 'M2'): 3, ('A', 'M3'): 10,
        ('B', 'M1'): 5, ('B', 'M2'):10, ('B', 'M3'): 10,
        ('C', 'M1'): 5, ('C', 'M2'): 3, ('C', 'M3'): 9,
        ('D', 'M1'): 4, ('D', 'M2'): 6, ('D', 'M3'): 3,
        ('E', 'M1'): 8, ('E', 'M2'): 2, ('E', 'M3'): 8,
        ('F', 'M1'): 5, ('F', 'M2'): 2, ('F', 'M3'): 10,
        ('G', 'M1'): 3, ('G', 'M2'): 2, ('G', 'M3'): 7
    }

    model.p_material = pyo.Param(model.products, model.materials, initialize=required_material)

    # objective
    model.obj = pyo.Objective(expr=sum(model.p_quantity[p] * model.prices[p] for p in model.products), sense=pyo.maximize)

    # constraints
    material_limits = {'M1': 100, 'M2': 150, 'M3': 200}
    model.material_limit = pyo.Param(model.materials, initialize=material_limits)

    def material_constraints(model, m):
        return sum(model.p_quantity[p] * model.p_material[p, m] for p in model.products) <= model.material_limit[m]

    model.material_constraint = pyo.Constraint(model.materials, rule=material_constraints)

    result = solver.solve(model)

    print(f"Status = {result.solver.status}")
    print(f"Termination condition = {result.solver.termination_condition}")

    print("Optimal quantities: ")
    for p in model.products:
        print(f"{pyo.value(model.p_quantity[p])}")

    print(f"Optimal objective = {pyo.value(model.obj)}")


def linear_regression(model, X, y):

    solver = pyo.SolverFactory('ipopt')

    observations = [i for i in range(len(X))]
    model.observations = pyo.Set(initialize=observations)

    x_dict = {observations[i]: X[i] for i in range(len(observations))}
    model.x = pyo.Param(model.observations, initialize=x_dict)

    y_dict = {observations[i]: y[i] for i in range(len(observations))}
    model.y = pyo.Param(model.observations, initialize=y_dict)

    model.alpha = pyo.Var()
    model.beta = pyo.Var()

    model.obj = pyo.Objective(
        expr=sum((y_dict[i] - (model.alpha + model.beta * x_dict[i])) ** 2 for i in range(len(observations))),
        sense=pyo.minimize
    )

    result = solver.solve(model)

    print(f"Status = {result.solver.status}")
    print(f"Termination condition = {result.solver.termination_condition}")

    print(f"alpha = {pyo.value(model.alpha)}")
    print(f"beta = {pyo.value(model.beta)}")

    print(f"objective function = {pyo.value(model.obj)}")


if __name__ == '__main__':

    model = pyo.ConcreteModel()
    # product_mix()

    X = np.array([38, 56, 50, 52, 37, 60, 67, 54, 59, 43, 30, 53, 59, 40, 65])
    y = np.array([137, 201, 152, 107, 150, 173, 194, 166, 154, 137, 38, 193, 153, 175, 247])

    linear_regression(model, X, y)

