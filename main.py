from datasetgenerator import DatasetGenerator
import numpy as np

dsg = DatasetGenerator()
length, nodes, stations, chargers = dsg.generate_highway(40)
car_types_battery, car_types_consumptions = dsg.generate_car_types(2)
selected_cars, selected_feasible_solution = dsg.generate_car_instances_density(10.0)
a = 1