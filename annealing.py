import pandas as pd
import numpy as np
from pyqubo import Binary, Base
from neal import SimulatedAnnealingSampler

def instance_to_matrix(instance, nodes, stations):
    speeds = pd.DataFrame({'speed': [60,70,80,90,100,110]})
    instance['original'] = instance.index
    im = instance.copy(deep=True)
    im['entry_node'] = nodes[im['entry_node_index']]
    im['exit_node'] = nodes[im['exit_node_index']]
    im = im.merge(speeds, how='cross').merge(pd.DataFrame({'station':stations}), how='cross')
    im['time_of_arrival'] = im['entry_timestamp'] + ( (im['station'] - im['entry_node']) / im['speed'] )
    collision_matrix = np.zeros((im.shape[0], im.shape[0]))
    #TODO: check if it is possible for a car to go to that station
    #TODO: add multiple chargers in each station
    for index1, row1 in im.iterrows():
        for index2, row2 in im.iterrows():
            if index2 <= index1:
                continue
            if abs(row1['time_of_arrival'] - row2['time_of_arrival']) < 100:
                collision_matrix[index1][index2] = collision_matrix[index2][index1] = 1

    return collision_matrix


def matrix_to_bqm(cars: pd.DataFrame, conflict_matrix: np.ndarray):
    conflict_strength = 1
    onehot_strength = 1
    H: Base = 0
    variables = {}
    for id in cars.index:
        variables[id] = Binary(str(id))

    # only one option for each car
    for _, group in cars.groupby('original'):
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

def solve_bqm(bqm):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=1000)
    first_sample = sampleset.first.sample
    print(first_sample)
