import math
from typing import Any

import pyomo.environ as pyo

def print_solver_info(result):
    print(f"Status = {result.solver.status}")
    print(f"Termination condition = {result.solver.termination_condition}")

def preprocess_jobs(conflicting_jobs: list[list[int]]) -> dict:

    arr = [[1 for _ in range(len(conflicting_jobs))] for _ in range(len(conflicting_jobs))]

    for i in range(len(conflicting_jobs)):
        inner_list = conflicting_jobs[i]
        if inner_list:
            for j in inner_list:
                arr[i][j] = 0

    final_dict = {
        (i, j): arr[i][j]
        for i in range(len(arr))
        for j in range(len(arr))
    }

    return final_dict

# P1
def parallel_machine_scheduling(model: pyo.ConcreteModel, processing_times: list[int], conflicting_jobs: list[list[int]],
                                num_machines: int) -> int:

    if len(processing_times) != len(conflicting_jobs):
        print("The ratio between the number of jobs and the size of the conflicting jobs should be 1:1")
        return -1

    conflicting_jobs = preprocess_jobs(conflicting_jobs)

    solver = pyo.SolverFactory('glpk') # solver for LPs and IPs

    # jobs
    job_indexes = [i for i in range(len(processing_times))]
    model.J = pyo.Set(initialize=job_indexes)

    # machines
    machine_indexes = range(num_machines)
    model.M = pyo.Set(initialize=machine_indexes)

    # decision variables
    model.x = pyo.Var(model.J, model.M, domain=pyo.Binary)

    # parameters
    p_times_dict = {i : processing_times[i] for i in range(len(job_indexes))}
    model.p_times = pyo.Param(model.J, initialize=p_times_dict)

    model.w = pyo.Param(model.J, model.J, initialize=conflicting_jobs)

    # objective
    model.t = pyo.Var(domain=pyo.NonNegativeReals)  # the makespan
    model.obj = pyo.Objective(expr=model.t, sense=pyo.minimize)

    # constraints
    def assignment_constraints(model, i):
        return sum(model.x[i, m] for m in model.M) == 1

    # constraints must be indexed over a Set
    model.assignment_constraint = pyo.Constraint(model.J, rule=assignment_constraints)

    def makespan_constraints(model, m):
        return sum(model.p_times[i] * model.x[i, m] for i in model.J) <= model.t

    model.makespan_constraint = pyo.Constraint(model.M, rule=makespan_constraints)

    def conflicting_constraints(model, i, j, m):
        if model.w[i, j] == 0:
            return model.x[i, m]  + model.x[j, m] <= 1
        return pyo.Constraint.Skip

    model.conflicting_constraint = pyo.Constraint(model.J, model.J, model.M, rule=conflicting_constraints)

    # result
    result = solver.solve(model)
    print_solver_info(result)

    print(f"Optimal makespan/objective = {pyo.value(model.obj)}")  # model.t == model.obj

    return pyo.value(model.obj)


# P2
def min_max_facility_location(model: pyo.ConcreteModel, distances: list[list[int]], district_population: list[int],
                              num_facilities: int) -> int:

    solver = pyo.SolverFactory('glpk')

    if len(distances) != len(district_population):
        print("The ratio between the number of districts and the distances array should be 1:1")
        return -1

    # facilities
    model.N = pyo.Set(initialize=range(num_facilities))

    # districts
    model.M = pyo.Set(initialize=range(len(distances)))

    # decision variables
    model.x = pyo.Var(model.M, domain=pyo.Binary)
    model.y = pyo.Var(model.M, model.M, domain=pyo.Binary)

    # parameters
    distances_dict = {
        (i, j): distances[i][j]
        for i in range(len(distances))
        for j in range(len(distances))
    }
    model.distances = pyo.Param(model.M, model.M, initialize=distances_dict)

    population_dict = {i : district_population[i] for i in range(len(district_population))}
    model.population = pyo.Param(model.M, initialize=population_dict)

    # objective
    model.t = pyo.Var(domain=pyo.NonNegativeReals)

    # trick: introduce t, and then force t to be >= than all the individual metrics
    model.obj = pyo.Objective(expr=model.t, sense=pyo.minimize)

    # constraints

    # exactly N facilities allocated
    def c1_rule(model):
        return sum(model.x[j] for j in model.M) == len(model.N)
    model.c1 = pyo.Constraint(rule=c1_rule)

    # assign facilities only to open locations
    def c2_rule(model, i, j):
        return model.y[i, j] <= model.x[j]
    model.c2 = pyo.Constraint(model.M, model.M, rule=c2_rule)

    # only one facility can be the closest
    def c3_rule(model, i):
        return sum(model.y[i, j] for j in model.M) == 1
    model.c3 = pyo.Constraint(model.M, rule=c3_rule)

    def c4_rule(model, i): # the sum is after j, i is given
        return model.t >= (model.population[i] * sum(model.distances[i, j] * model.y[i, j] for j in model.M))
    model.c4 = pyo.Constraint( model.M, rule=c4_rule)

    result = solver.solve(model)
    print_solver_info(result)

    print(f"Found objective = {pyo.value(model.t)}")
    return pyo.value(model.obj)


# P3
def compute_objective(districts: list[int], located_facilities: list[int], distances: list[list[int]],
                      district_population: list[int]) -> int:

    weighted_distance = [0] * len(districts)
    closest_facilities = [0] * len(districts)
    print(f"located_facilities = {located_facilities}")

    if len(located_facilities) == 1:
        closest_facilities = [located_facilities[0]] * len(districts)
    else:
        for i in range(len(districts)):
            closest_facility_idx = 0

            for j in range(len(located_facilities)):
                located_facility = located_facilities[j]
                if distances[i][located_facilities[closest_facility_idx]] > distances[i][located_facility]:
                    closest_facility_idx = j

            closest_facilities[i] = located_facilities[closest_facility_idx]

    for i in range(len(districts)):
        weighted_distance[i] += district_population[i] * distances[i][closest_facilities[i]]

    print(f"weighted_distance = {weighted_distance}")
    return max(weighted_distance)


def choose_district(districts: list[int], distances: list[list[int]], district_population: list[int]) -> tuple[int, int]:

    empty_districts = []
    located_facilities = []
    for i in range(len(districts)):
        if districts[i] == 0:
            empty_districts.append(i)
        elif districts[i] == 1:
            located_facilities.append(i)

    temp_arr = located_facilities[:] # temp arr - where we may consider to add another facility
    best_district = 0
    min_val = math.inf

    for i in range(len(empty_districts)):
        temp_arr.append(empty_districts[i])
        current_objective = compute_objective(districts, temp_arr, distances, district_population)

        if current_objective < min_val: # choose greedily the district with the lowest weighted distance
            min_val = current_objective
            best_district = empty_districts[i]
        temp_arr = located_facilities[:] # reset the available facilities

    return best_district, min_val


def heuristic_facility_allocation(distances: list[list[int]], district_population: list[int], num_facilities: int) -> int:

    """
    Args:
        distances: a 2d array containing the distance between every 2 districts
        district_population: an array containing the population (in thousands)
        num_facilities: how many facilities we may allocate

    Returns: the minimized maximum population-weighted firefighting (PWFT) times among all districts

    idea: For a given number of iterations, equal to num_facilities, locate an ambulance in a district that:
            (1) currently does not have an ambulance AND
            (2) minimizes PWFT among all districts
          If there are multiple districts satisfying these two conditions, pick the one with the smallest district ID.
    """

    if num_facilities >= len(distances):
        min_val = 0
    else:
        districts = [0] * len(district_population)
        num_iterations = 0
        while num_iterations < num_facilities:
            print(f"Now at iteration {num_iterations + 1}")

            idx, min_val = choose_district(districts, distances, district_population)
            print(f"Chosen idx = {idx}")

            districts[idx] = 1
            num_iterations += 1

    print(f"Objective by heuristic = {min_val}")
    return min_val

if __name__ == '__main__':

    p_model = pyo.ConcreteModel()

    """
    p_times = [7, 4, 6, 9, 12, 8, 10, 11, 8, 7, 6, 8, 15, 14, 3]
    c_jobs = [[], [4, 7], [], [], [1, 7], [8], [9], [1, 4], [5], [6], [14], [], [], [], [10]]
    parallel_machine_scheduling(model=p_model, processing_times=p_times, conflicting_jobs=c_jobs, num_machines=3)
    """

    p_distances = [[0, 3, 4, 6, 8, 9, 8, 10],
                   [3, 0, 5, 4, 8, 6, 12, 9],
                   [4, 5, 0, 2, 2, 3, 5, 7],
                   [6, 4, 2, 0, 3, 2, 5, 4],
                   [8, 8, 2, 3, 0, 2, 2, 4],
                   [9, 6, 3, 2, 2, 0, 3, 2],
                   [8, 12, 5, 5, 2, 3, 0, 2],
                   [10, 9, 7, 4, 4, 2, 2, 0]]

    p_district_population = [40, 30, 35, 20, 15, 50, 45, 60]

    # num_facilities = 2 ,  num_facilities = 3
    # optimal_obj  = 135,   optimal_obj  = 100
    # min_max_facility_location(model=p_model, distances=p_distances, district_population=p_district_population, num_facilities=3)

    # num_facilities = 2 ,  num_facilities = 3
    # optimal_obj  = 240,   optimal_obj  = 100
    heuristic_facility_allocation(distances=p_distances, district_population=p_district_population, num_facilities=3)
