import pandas as pd
import numpy as np
from pyqubo import Binary, Base
from neal import SimulatedAnnealingSampler
from datasetgenerator import DatasetGenerator
from dwave.system import DWaveSampler, LeapHybridSampler, LeapHybridCQMSampler
import dimod

def instance_to_matrix(instance,car_info, length, nodes, stations, chargers, available_car_speeds, charging_speed):
    '''dsg = DatasetGenerator(max_length=500, available_car_speeds=[60, 70], charging_speed=22)
    length, nodes, stations, chargers = dsg.generate_highway(100)
    car_info = dsg.generate_car_types(2)
    instance = dsg.generate_car_instances(10.0, 0.1)'''

    speeds = pd.DataFrame({'speed': available_car_speeds})
    stations = pd.DataFrame({'station_index': range(len(stations)), 'station_kilometer':stations})
    instance['original'] = instance.index

    im = instance.copy(deep=True)
    im['entry_kilometer'] = nodes[im['entry_node_index']]
    im['exit_kilometer'] = nodes[im['exit_node_index']]
    im = im.merge(speeds, how='cross').merge(stations, how='cross')

    im = (im[im['entry_kilometer'] <= im['station_kilometer']]).reset_index(drop=True)

    im['station_timestamp'] = ((im['entry_timestamp']/3600 + ((im['station_kilometer'] - im['entry_kilometer']) / im['speed'] )) * 3600).round().astype(int)
    burn = pd.Series(car_info.lookup(im['car_type'],im['speed']))
    im['station_energy'] = car_info['capacity'][im['car_type']].reset_index(drop=True) * im['entry_energy']
    im['station_energy'] -= (burn * (im['station_kilometer'] - im['entry_kilometer']) / 1000)
    im['station_energy'] /= car_info['capacity'][im['car_type']].reset_index(drop=True)
    im['charging_time'] = (1 - im['station_energy']) * car_info['capacity'][im['car_type']].reset_index(drop=True) / charging_speed * 3600
    im['max_kilometers'] = car_info['capacity'][im['car_type']].reset_index(drop=True)*im['entry_energy'] / (pd.Series(car_info.lookup(im['car_type'],im['speed']))/1000)
    im = im[im['station_energy'] > 0.].reset_index(drop=True)

    collision_matrix = np.zeros((im.shape[0], im.shape[0]), dtype=int)

    for index1, row1 in im.iterrows():
        for index2, row2 in im.iterrows():
            if index2 <= index1:
                continue
            if row1['station_index'] != row2['station_index']:
                continue
            # first to come should be row1
            if row2['station_timestamp'] >= row1['station_timestamp']:
                if row2['station_timestamp'] - row1['station_timestamp'] < row1['charging_time']:
                    collision_matrix[index1][index2] = collision_matrix[index2][index1] = 1
            else:
                if row1['station_timestamp'] - row2['station_timestamp'] < row2['charging_time']:
                    collision_matrix[index1][index2] = collision_matrix[index2][index1] = 1

    return collision_matrix, im


def matrix_to_bqm(cars: pd.DataFrame, conflict_matrix: np.ndarray):
    conflict_strength = 1
    onehot_strength = 2
    H: Base = 0
    variables = {}
    for id in cars.index:
        variables[id] = Binary(str(id))

    # one-hot
    for _, group in cars.groupby('original'):
        H_term = 0
        for id in group.index:
            H_term += variables[id]
        H += ((1-H_term)**2) * onehot_strength
    # stations chargers
    for id1 in cars.index:
        for id2 in cars.index:
            # is upper-triangular matrix guaranteed or should we check both triangles?
            if id2 <= id1:
                continue
            if conflict_matrix[id1][id2]:
                H += variables[id1] * variables[id2] * conflict_strength

    model = H.compile()
    return model.to_bqm()


def matrix_to_cqm(im: pd.DataFrame, collision_matrix: np.ndarray, chargers):
    cqm = dimod.ConstrainedQuadraticModel()
# cqm.set_objective
    variables = {}
    for id1 in im.index:
        variables[id1] = dimod.Binary(str(id1))

    # one-hot
    for car_id, group in im.groupby('original'):
        one_car_vars = [variables[id1] for id1 in group.index]
        cqm.add_constraint(sum(one_car_vars) == 1, label=f'One-hot constraint of car {car_id}.')
    # stations chargers
    for id1 in im.index:
        one_station_vars = [variables[id1]]
        for id2 in im.index:
            if id2 <= id1:
                continue
            if collision_matrix[id1][id2]:
                one_station_vars.append(variables[id2])
        cqm.add_constraint(sum(one_station_vars) <= chargers[im['station_index'][id1]], label=f"limit number of cars on station {im['station_index'][id1]} for car {id1}.")

    return cqm


def bqm_to_sampleset(bqm):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=10000)
    return sampleset

def cqm_to_sampleset(cqm):
    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm, time_limit=100)
    return sampleset
