import pandas as pd
import numpy as np
from pyqubo import Binary, Base
from neal import SimulatedAnnealingSampler

def matrix_to_bqm(cars: pd.DataFrame, conflict_matrix: np.ndarray):
    conflict_strength = 1
    onehot_strength = 1
    H: Base = 0
    variables = {}
    for id in cars.index:
        variables[id] = Binary(id)

    # only one option for each car
    for _, group in cars.groupby('generated_from'):
        H_term = 0
        for id in group.index:
            H_term += variables[id]
        H += ((1-H_term)**2) * onehot_strength

    for i, idi in enumerate(cars.index):
        for j, idj in enumerate(cars.index):
            # is upper-triangular matrix guaranteed or should we check both triangles?
            if j <= j:
                continue
            if conflict_matrix[i][j]:
                H += variables[idi] * variables[idj] * conflict_strength

    model = H.compile()
    return model.to_bqm()

def solve_qubo(bqm):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=1000)
    decoded_sampleset = sampleset.decode_sampleset(sampleset)
    print(decoded_sampleset)
