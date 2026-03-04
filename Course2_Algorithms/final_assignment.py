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
def parallel_machine_scheduling(model: pyo.ConcreteModel, num_machines: int, processing_times: list[int],
                                conflicting_jobs: list[list[int]]) -> int | None:

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


# P2

def facility_location(model: pyo.ConcreteModel) -> None:

if __name__ == '__main__':

    p_model = pyo.ConcreteModel()

    p_times = [7, 4, 6, 9, 12, 8, 10, 11, 8, 7, 6, 8, 15, 14, 3]
    c_jobs = [[], [4, 7], [], [], [1, 7], [8], [9], [1, 4], [5], [6], [14], [], [], [], [10]]
    parallel_machine_scheduling(model=p_model, num_machines=3, processing_times=p_times, conflicting_jobs=c_jobs)