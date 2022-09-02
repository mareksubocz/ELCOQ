import numpy as np

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
            n_nodes = round(length / node_freq) + 2
        elif n_nodes < 2:
            raise ValueError('Number of nodes must not be less than 2')

        if n_stations is None:
            n_stations = round(length / station_freq) + 1

        elif n_stations < 1:
            raise ValueError('Number of stations must not be less than 1')

        nodes = np.sort(np.random.randint(1, length, n_nodes-2))
        nodes = np.append(nodes, length)
        self.nodes = np.insert(nodes, 0, 0)

        self.stations = np.sort(np.random.randint(1, length, n_stations))
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

        return self.car_types_battery, self.car_types_consumptions

        #TODO make sure that cars can travel the motorway with only one charging

    def generate_car_instances_density(self, density=None):
        sum_c = np.sum(self.chargers)
        n_cars = round(sum_c * density)
        if n_cars < 1:
            raise ValueError('Too low density, make it bigger')
        return self.generate_car_instances(n_cars)


    def generate_car_instances(self, n_cars):
        if n_cars < 1:
            raise ValueError('There must be at least one car')
        max_battery_size = max(self.car_types_battery)
        min_speed = min(self.available_car_speeds)

        simulation_time = self.length/min_speed + max_battery_size/self.charging_speed
        simulation_time_seconds = round(simulation_time * 60 * 60)
        car_charging_times = np.ceil((np.asarray(self.car_types_battery) / self.charging_speed) * 60 * 60).astype(int)
        car_max_distances = np.asarray(self.car_types_battery) / np.min(self.car_types_consumptions, axis=1) * 1000

        cars = []
        feasible_solution = []
        #for car in range(n_cars):
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
                    started_with = np.ceil((consumed_percent + (1-consumed_percent) * np.random.random())*100)/100
                    second_to_charge = np.ceil(car_charging_times[car_type] * (started_with - consumed_percent))
                    current_time += second_to_charge

                    car_start_time = current_time - np.ceil(diff_entry_station/self.available_car_speeds[0]*60*60)
                    possible_out_nodes = np.logical_and(nodes_station_distance < car_max_distance, nodes_station_distance > 0)
                    out_node = np.random.choice(np.arange(len(self.nodes))[possible_out_nodes])

                    cars.append([car_type[0], entry_node, car_start_time[0], out_node, started_with[0]])
                    feasible_solution.append(i)

        cars = np.asarray(cars)
        min_timestamp = min(cars[:, 2])
        cars[:, 2] += abs(min_timestamp)
        feasible_solution = np.asarray(feasible_solution)
        selected_cars_indices = np.random.choice(np.arange(len(cars)), n_cars)
        selected_cars = cars[selected_cars_indices]
        selected_feasible_solution = feasible_solution[selected_cars_indices]
        return selected_cars, selected_feasible_solution




a = 1