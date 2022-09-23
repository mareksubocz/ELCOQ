from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
import pandas as pd

import numpy as np
from itertools import combinations, groupby
from datasetgenerator import DatasetGenerator
from annealing import instance_to_matrix
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp, DictStateFn

#from qiskit.opflow.state_fns import OperatorStateFn

def get_collisions(collision_matrix, im, chargers):
    collision_matrix_filled_diagonal = np.copy(collision_matrix)
    np.fill_diagonal(collision_matrix_filled_diagonal, 1)

    conflicts = []
    for station_id in range(len(chargers)):
        station_mask = im['station_index'] == station_id
        ims = im[station_mask]
        cms = collision_matrix_filled_diagonal[station_mask][:, station_mask]
        for i, index in enumerate(ims.index):
            row = cms[i]
            car_conflict_mask = row == 1
            conflicted_with_car = ims[car_conflict_mask]['original']
            conflicted_with_car = conflicted_with_car[(conflicted_with_car != (im.iloc[index]['original'])) ^ (conflicted_with_car.index == index)]
            all_conflicts = list(combinations(conflicted_with_car.index, chargers[station_id] + 1))
            filtered_conflicts = all_conflicts.copy()

            for j in conflicted_with_car.unique():
                duplicates = list(conflicted_with_car[conflicted_with_car == j].index)
                if len(duplicates) > 1:
                    for e in all_conflicts:
                        if set(duplicates).issubset(e):
                            filtered_conflicts.remove(e)
            conflicts.extend(filtered_conflicts)

    conflicts.sort()
    conflicts = list(conflict for conflict, _ in groupby(conflicts))
    return conflicts

def get_onehot(im):
    return [list(v) for _, v in im.groupby('original').indices.items()]

def get_conflict_hamiltonian(conflicts, n):
    conflict_hamiltonian = None
    for c in conflicts:
        product = None
        for i in c:
            sparse_list = []
            sparse_list.append((('I', [i], 1)))
            sparse_list.append((('Z', [i], -1)))
            sp = SparsePauliOp.from_sparse_list(sparse_list, n)
            if product is None:
                product = sp
            else:
                product = product.compose(sp)
        product *= 1 / (2 ** len(c))
        if conflict_hamiltonian is None:
            conflict_hamiltonian = product
        else:
            conflict_hamiltonian += product
    conflict_hamiltonian = PauliSumOp(conflict_hamiltonian).reduce()
    return conflict_hamiltonian


def get_onehot_hamiltonian(onehot, n):
    onehot_hamiltonian = None
    for oh in onehot:
        product = None
        for i in oh:
            sp = SparsePauliOp.from_sparse_list([(('Z', [i], 1))], n)
            if product is None:
                product = sp
            else:
                product = product.compose(sp)
        product = (SparsePauliOp.from_sparse_list([(('I', [i], 1))], n) + product) / 2
        if onehot_hamiltonian is None:
            onehot_hamiltonian = product
        else:
            onehot_hamiltonian += product
    onehot_hamiltonian = PauliSumOp(onehot_hamiltonian).reduce()
    return onehot_hamiltonian


available_car_speeds = [60, 100]

dsg = DatasetGenerator(max_length=500, available_car_speeds=available_car_speeds, charging_speed=22, seed=5)
length, nodes, stations, chargers = dsg.generate_highway(30)
car_info = dsg.generate_car_types(2)
instance = dsg.generate_car_instances(10.0, 0.1)

collision_matrix, im = instance_to_matrix(instance,car_info, length, nodes, stations, chargers, available_car_speeds, charging_speed=22)
conflicts = get_collisions(collision_matrix, im, chargers)
onehot = get_onehot(im)

conflict_hamiltonian = get_conflict_hamiltonian(conflicts, len(im))
onehot_hamiltonian = get_onehot_hamiltonian(onehot, len(im))

hamiltonian = (conflict_hamiltonian + onehot_hamiltonian).reduce()

optimizer = COBYLA()


for p in range(1, 6):
    prob_zero_mean = 0
    for i in range(10):
        qaoa = QAOA(optimizer, quantum_instance=Aer.get_backend('aer_simulator'), reps=p)
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        keys, values = zip(*result.eigenstate.items())
        energies = abs(np.asarray(list(hamiltonian.eval(DictStateFn({k: 1 for k, _ in result.eigenstate.items()})).primitive.values())))
        res_df = pd.DataFrame({'binary': keys, 'prob_squared': values, 'energy': energies})
        prob_zero = sum(res_df[res_df['energy'] == 0]['prob_squared'] ** 2)
        print(prob_zero)
        prob_zero_mean += prob_zero
    print('p: {}, prob_zero: {}'.format(p, prob_zero_mean / 10))

#{k: abs(v) for k, v in hamiltonian.eval(DictStateFn({k: v**2 for k, v in result.eigenstate.items()})).primitive.items()}
a = 1