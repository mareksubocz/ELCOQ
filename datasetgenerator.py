import numpy as np

class DatasetGenerator:
    def __init__(self):
        pass

    def generate_highway(self, length=None, n_nodes=None, n_stations=None):

        if length is None:
            length = np.random.randint(2, 500)
        elif length < 2:
            raise ValueError('Length must not be less than 2 km')

        if n_nodes is None:
            n_nodes = round(length / 15) + 2
        elif n_nodes < 2:
            raise ValueError('Number of nodes must not be less than 2')

        if n_stations is None:
            n_stations = round(length / 50) + 1

        elif n_stations < 1:
            raise ValueError('Number of stations must not be less than 1')

        nodes = np.random.randint(1, length, n_nodes-2)
        nodes = np.append(nodes, length)
        nodes = np.insert(nodes, 0, 0)

        stations = np.random.randint(1, length, n_stations)

        return length, nodes, stations


dsg = DatasetGenerator()
length, nodes, stations = dsg.generate_highway(2)