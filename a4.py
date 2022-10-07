import numpy as np
from datasetgenerator import DatasetGenerator
from datetime import datetime
import utils

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

instance, feasible_solution = dsg.generate_car_instances(1000000, 0.5)