from datasetgenerator import DatasetGenerator
import numpy as np

dsg = DatasetGenerator(max_length=500, available_car_speeds=[60, 70, 80, 90, 100, 110], charging_speed=22)
length, nodes, stations, chargers = dsg.generate_highway(100)
car_info = dsg.generate_car_types(2)
instance = dsg.generate_car_instances(10.0, 0.1)