from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit import Aer
import pandas as pd

import numpy as np
from itertools import combinations, groupby
from datasetgenerator import DatasetGenerator
from annealing import instance_to_matrix
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp, DictStateFn
from datetime import datetime

from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

from qiskit.providers.fake_provider import FakeKolkataV2

from qiskit_ibm_runtime import QiskitRuntimeService

import utils

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


logdir = 'logdir' + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
instance_name = 'small_instance'
data_path = 'data/{}'.format(instance_name)

'''available_car_speeds = [60, 100]
dsg = DatasetGenerator(max_length=500, available_car_speeds=available_car_speeds, charging_speed=22, seed=5)
length, nodes, stations, chargers = dsg.generate_highway(30)
car_info = dsg.generate_car_types(2)
instance, feasible_solution = dsg.generate_car_instances(10.0, 0.1)
instance_info = {'available_car_speeds': available_car_speeds,
                'length': length,
                'nodes': nodes,
                'stations': stations,
                'chargers': chargers,
                'feasible_solution': feasible_solution}

utils.save_instance_data(car_info, instance_info, instance, data_path)'''
car_info, instance_info, instance = utils.load_instance_data(data_path)
length = instance_info['length']
nodes = instance_info['nodes']
stations = instance_info['stations']
chargers = instance_info['chargers']
available_car_speeds = instance_info['available_car_speeds']

collision_matrix, im = instance_to_matrix(instance,car_info, nodes, stations, available_car_speeds, charging_speed=22)
conflicts = get_collisions(collision_matrix, im, chargers)
onehot = get_onehot(im)

conflict_hamiltonian = get_conflict_hamiltonian(conflicts, len(im))
onehot_hamiltonian = get_onehot_hamiltonian(onehot, len(im))

hamiltonian = (conflict_hamiltonian + onehot_hamiltonian).reduce()

#optimizer = COBYLA()
optimizer = SPSA()

IBMQ.load_account()
#provider = IBMQ.get_provider('ibm-q-psnc', 'internal', 'reservations')
provider = IBMQ.get_provider('ibm-q-psnc', 'internal', 'default')
backend = provider.get_backend('ibmq_kolkata')
#backend = FakeKolkataV2()
providers = [x for x in provider.backends() if 'simulator' not in x.name()]

options = {
	#'backend_name': 'ibmq_toronto'
    #'backend_name': 'ibm_hanoi',
    'backend_name': 'ibmq_mumbai'
    #'backend_name': least_busy(providers).name()
}

from qiskit_ibm_runtime import QiskitRuntimeService

# Save your credentials on disk.
#QiskitRuntimeService.save_account(channel='ibm_quantum', token='1e637c6bc70c02597cfc4f76b572766876969710b364e11205257964a1faa557e2fba978074ce75befd5ede100b73e06cde673d7c872257ae81b03e91bf93a06')

'''service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q-psnc/internal/default',
)'''

service = QiskitRuntimeService(
	channel='ibm_quantum'
)

for p in range(1, 6):

    prob_zero_mean = 0
    runtime_inputs = {
        'optimizer': optimizer,
        'operator': hamiltonian,
        'use_swap_strategies': False,
        'optimization_level': 3,
        #'measurement_error_mitigation': True,
        #'use_initial_mapping': True,
        #'use_pulse_efficient': True,
        'reps': p
    }

    for i in range(1):
        #qaoa = QAOA(optimizer, quantum_instance=Aer.get_backend('aer_simulator'), reps=p)
        #qaoa = QAOA(optimizer, quantum_instance=backend, reps=p)
        #result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        #eigenstate = result.eigenstate

        job = service.run(
            program_id='qaoa',
            options=options,
            inputs=runtime_inputs
        )
        result = job.result()
        eigenstate = result['eigenstate']

        keys, values = zip(*eigenstate.items())
        energies = abs(np.asarray(list(hamiltonian.eval(DictStateFn({k: 1 for k, _ in eigenstate.items()})).primitive.values())))
        res_df = pd.DataFrame({'binary': keys, 'prob_squared': values, 'energy': energies})
        prob_zero = sum(res_df[res_df['energy'] == 0]['prob_squared'] ** 2)
        print(prob_zero)
        prob_zero_mean += prob_zero

        energy = sum(res_df['energy']*res_df['prob_squared']**2)
        utils.save_true_quantum_result(res_df, logdir, instance_name, p, energy)
    print('p: {}, prob_zero: {}'.format(p, prob_zero_mean / 10))




#{k: abs(v) for k, v in hamiltonian.eval(DictStateFn({k: v**2 for k, v in result.eigenstate.items()})).primitive.items()}
a = 1