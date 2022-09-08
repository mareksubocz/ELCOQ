import numpy as np
import pandas as pd

class DatasetGenerator:
    def __init__(self, max_length=500, available_car_speeds=[60, 70, 80, 90, 100, 110], charging_speed=22):
        self.max_length = max_length
        self.available_car_speeds = available_car_speeds
        self.charging_speed = charging_speed

    def generate_highway(self, length=None, n_nodes=None, node_freq=15, n_stations=None, station_freq=25, n_chargers_min=2, n_chargers_max=3):

        if length is None:
            self.length = np.random.randint(2, 500)
        elif length < 2:
            raise ValueError('Length must not be less than 2 km')
        else:
            self.length = length

        if n_nodes is None:
            n_nodes = round(self.length / node_freq) + 2
        elif n_nodes < 2:
            raise ValueError('Number of nodes must not be less than 2')

        if n_stations is None:
            n_stations = round(self.length / station_freq) + 1

        elif n_stations < 1:
            raise ValueError('Number of stations must not be less than 1')

        nodes = np.sort(np.random.randint(1, self.length, n_nodes-2))
        nodes = np.append(nodes, self.length)
        self.nodes = np.insert(nodes, 0, 0)

        self.stations = np.sort(np.random.randint(1, self.length, n_stations))
        self.chargers = np.random.randint(n_chargers_min, n_chargers_max+1, n_stations)

        return self.length, self.nodes, self.stations, self.chargers

    def generate_car_types(self, n, min_kVh=10, max_kWh=30, min_consumption=80, max_consumption=250):

        if n < 1:
            raise ValueError('There must be at least one car type')

        if min_kVh < 5:
            raise ValueError('Minimum battery size must be at least 5')

        if min_kVh >= max_kWh:
            raise ValueError('Maximum battery size must be greater than minimum battery size')

        if min_consumption < 1:
            raise ValueError('Minimum consumption size must be at least 1')
        else:
            self.min_consumption = min_consumption

        if min_consumption >= max_consumption:
            raise ValueError('Maximum consumption must be greater than minimum consumption')
        else:
            self.max_consumption = max_consumption

        self.car_types_battery = np.random.randint(min_kVh, max_kWh + 1, n)
        self.car_types_consumptions = []
        for _ in self.car_types_battery:
            car_consumption = np.random.randint(min_consumption, max_consumption+1, len(self.available_car_speeds))
            self.car_types_consumptions.append(np.sort(car_consumption))

        self.car_info = pd.DataFrame(np.hstack([self.car_types_battery.reshape(-1, 1), np.asarray(self.car_types_consumptions)]), columns=['capacity'] + list(self.available_car_speeds))
        return self.car_info

    def generate_car_instances(self, density=3.0, sim_length=1.0, n_cars=None, sim_seconds=None):

        max_battery_size = max(self.car_types_battery)
        min_speed = min(self.available_car_speeds)

        if sim_seconds is not None:
            simulation_time_seconds = sim_seconds
        else:
            simulation_time = (self.length/min_speed + max_battery_size/self.charging_speed) * sim_length
            simulation_time_seconds = round(simulation_time * 60 * 60)

        cars, feasible_solution = self._generate_cars(simulation_time_seconds)

        if n_cars is not None:
            if n_cars < 1:
                raise ValueError('There must be at least one car')
            n_to_choose = min(n_cars, len(cars))
        else:
            sum_c = np.sum(self.chargers)
            n_cars = round(sum_c * density)
            n_to_choose = min(n_cars, len(cars))

        selected_cars_indices = np.random.choice(np.arange(len(cars)), n_to_choose, replace=False)
        selected_cars = cars[selected_cars_indices]
        selected_feasible_solution = feasible_solution[selected_cars_indices]

        self.selected_cars = selected_cars
        self.selected_feasible_solution = selected_feasible_solution
        columns = [('car_type', int),
                   ('entry_node_index', int),
                   ('entry_timestamp', int),
                   ('exit_node_index', int),
                   ('entry_energy', float)
                   ]
        instance = pd.DataFrame(selected_cars, columns=[x[0] for x in columns])
        self.instance = instance.astype(dtype={x:y for (x, y) in columns})

        return self.instance


    def _generate_cars(self, simulation_time_seconds):
        if simulation_time_seconds < 600:
            raise ValueError('Simulation time must be greater than 600 seconds')

        car_charging_times = np.ceil((np.asarray(self.car_types_battery) / self.charging_speed) * 60 * 60).astype(int)
        car_max_distances = np.asarray(self.car_types_battery) / np.min(self.car_types_consumptions, axis=1) * 1000

        cars = []
        feasible_solution = []
        for i, station in enumerate(self.stations):
            for charger in range(self.chargers[i]):
                current_time = 0
                while current_time < simulation_time_seconds:
                    car_type = np.random.randint(0, len(self.car_types_battery), 1)
                    car_max_distance = car_max_distances[car_type]
                    nodes_station_distance = self.nodes - station
                    possible_entry_nodes = np.logical_and(nodes_station_distance > -car_max_distance, nodes_station_distance < 0)
                    entry_node = np.random.choice(np.arange(len(self.nodes))[possible_entry_nodes])

                    diff_entry_station = station - self.nodes[entry_node]
                    consumed = diff_entry_station * np.min(self.car_types_consumptions, axis=1)[car_type] / 1000
                    consumed_percent = consumed/self.car_types_battery[car_type]
                    initial_energy = np.ceil((consumed_percent + (1-consumed_percent) * np.random.random())*100)/100
                    seconds_to_charge = np.ceil(car_charging_times[car_type] * (initial_energy - consumed_percent))
                    current_time += seconds_to_charge

                    car_start_time = current_time - np.ceil(diff_entry_station/self.available_car_speeds[0]*60*60)
                    possible_out_nodes = np.logical_and(nodes_station_distance < car_max_distance, nodes_station_distance > 0)
                    out_node = np.random.choice(np.arange(len(self.nodes))[possible_out_nodes])

                    cars.append([car_type[0], entry_node, car_start_time[0], out_node, initial_energy[0]])
                    feasible_solution.append(i)

        cars = np.asarray(cars)
        min_timestamp = min(cars[:, 2])
        cars[:, 2] += abs(min_timestamp)
        feasible_solution = np.asarray(feasible_solution)

        return cars, feasible_solution
