import pandas as pd
import numpy as np
from pyqubo import Binary, Base
from neal import SimulatedAnnealingSampler
from datasetgenerator import DatasetGenerator

def instance_to_matrix(instance, nodes, stations, charging_speed):
    dsg = DatasetGenerator(max_length=500, available_car_speeds=[60, 70], charging_speed=22)
    length, nodes, stations_before, chargers = dsg.generate_highway(100)
    car_info = dsg.generate_car_types(2)
    instance = dsg.generate_car_instances(10.0, 0.1)

    speeds = pd.DataFrame({'speed': dsg.available_car_speeds})
    stations = []
    for i, station in enumerate(stations_before):
        stations.extend([station]*chargers[i])
    stations = pd.DataFrame({'station_kilometer':stations})
    instance['original'] = instance.index

    im = instance.copy(deep=True)
    im['entry_kilometer'] = nodes[im['entry_node_index']]
    im['exit_kilometer'] = nodes[im['exit_node_index']]
    im = im.merge(speeds, how='cross').merge(stations, how='cross')
    im = im[im['entry_kilometer'] <= im['station_kilometer']].reset_index()
    im['station_timestamp'] = ((im['entry_timestamp']/3600 + ((im['station_kilometer'] - im['entry_kilometer']) / im['speed'] )) * 3600).round().astype(int)
    burn = pd.Series(car_info.lookup(im['car_type'],im['speed']))
    im['station_energy'] = car_info['capacity'][im['car_type']].reset_index(drop=True) * im['entry_energy']
    im['station_energy'] -= (burn * (im['station_kilometer'] - im['entry_kilometer']) / 1000)
    im['station_energy'] /= car_info['capacity'][im['car_type']].reset_index(drop=True)
    im['charging_time'] = (1 - im['station_energy']) * car_info['capacity'][im['car_type']].reset_index(drop=True) / dsg.charging_speed * 3600
    im['max_kilometers'] = car_info['capacity'][im['car_type']].reset_index(drop=True)*im['entry_energy'] / (pd.Series(car_info.lookup(im['car_type'],im['speed']))/1000)
    im = im[im['station_energy'] > 0.].reset_index(drop=True)

    collision_matrix = np.zeros((im.shape[0], im.shape[0]))
    #TODO: check if it is possible for a car to go to that station
    #TODO: add multiple chargers in each station
    for index1, row1 in im.iterrows():
        for index2, row2 in im.iterrows():
            if index2 <= index1:
                continue
            # first to come should be row1
            if row2['station_timestamp'] < row1['station_timestamp']:
                row1, row2 = row2, row1

            if row2['station_timestamp'] - row1['station_timestamp'] < row1['charging_time']:
                collision_matrix[index1][index2] = collision_matrix[index2][index1] = 1
    return collision_matrix, im

def matrix_to_bqm(cars: pd.DataFrame, conflict_matrix: np.ndarray):
    conflict_strength = 1
    onehot_strength = 2
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
            if j <= i:
                continue
            if conflict_matrix[i][j]:
                H += variables[idi] * variables[idj] * conflict_strength

    model = H.compile()
    return model.to_bqm()

def bqm_to_sampleset(bqm):
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=10000)
    return sampleset
