import numpy as np
from itertools import combinations, groupby
from datasetgenerator import DatasetGenerator
from annealing import instance_to_matrix

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

available_car_speeds = [60, 100]

dsg = DatasetGenerator(max_length=500, available_car_speeds=available_car_speeds, charging_speed=22, seed=5)
length, nodes, stations, chargers = dsg.generate_highway(30)
car_info = dsg.generate_car_types(2)
instance = dsg.generate_car_instances(10.0, 0.1)

collision_matrix, im = instance_to_matrix(instance,car_info, length, nodes, stations, chargers, available_car_speeds, charging_speed=22)
conflicts = get_collisions(collision_matrix, im, chargers)
onehot = get_onehot(im)
a = 1