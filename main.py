# from datasetgenerator import DatasetGenerator
# import annealing
# import numpy as np
# available_car_speeds = [60, 70, 80, 90, 100, 110]
# dsg = DatasetGenerator(max_length=500, available_car_speeds=available_car_speeds, charging_speed=22)
# length, nodes, stations, chargers = dsg.generate_highway(30)
# car_info = dsg.generate_car_types(2)
# instance = dsg.generate_car_instances(10.0, 0.1)
#
# collision_matrix, im = annealing.instance_to_matrix(instance,car_info, length, nodes, stations, chargers, available_car_speeds, charging_speed=22)
import numpy as np
from datasetgenerator import DatasetGenerator
from datetime import datetime
import utils
import annealing

logdir = 'logdir' + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
instance_name = 'a4'
data_path = 'data/{}'.format(instance_name)

car_info, instance_info, _ = utils.load_instance_data(data_path)
dsg = DatasetGenerator(max_length=instance_info['length'], available_car_speeds=instance_info['available_car_speeds'], charging_speed=22, seed=5)

dsg.length = instance_info['length']
dsg.nodes = np.asarray(instance_info['nodes'])
dsg.stations = np.asarray(instance_info['stations'])
dsg.chargers = np.asarray(instance_info['chargers'])
dsg.car_types_battery = car_info['capacity'].values
dsg.car_types_consumptions = car_info.values[:, 1:]

instance, feasible_solution = dsg.generate_car_instances(1000000, 0.1)

collision_matrix, im = annealing.instance_to_matrix(instance,car_info, instance_info['nodes'], instance_info['stations'], instance_info['available_car_speeds'], charging_speed=22)

bqm = annealing.matrix_to_bqm(im, collision_matrix)
sampleset = annealing.bqm_to_sampleset(bqm)

print(sampleset.first.sample)
print(sampleset.first.energy)

filtered_sample = [int(k) for k, v in sampleset.first.sample.items() if v == 1]
print(filtered_sample)
im.loc[filtered_sample]
