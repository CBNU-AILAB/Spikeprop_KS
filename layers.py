import numpy as np
import utils


class SP_Dense:
    def __init__(self, output_dim, weight_initializer):
        self.input_dim = -1
        self.output_dim = output_dim
        self.neurons = None
        self.connections = None
        self.n_terminals = 0
        self.weight_initializer = weight_initializer


    def generate_neurons(self):
        t_neurons = np.full((self.output_dim), -1, dtype=int)
        self.neurons = t_neurons


    def generate_connections(self):
        w = np.random.random((self.input_dim, self.output_dim, self.n_terminals, 1))

        y = np.zeros((self.input_dim, self.output_dim, self.n_terminals, 1))

        t_connections = np.concatenate((y, w), axis=3)

        self.connections = t_connections


    def build(self):
        is_built = False
        # make a neurons and connections between previous layer and added layer.
        self.generate_neurons()
        if self.input_dim == -1:
            is_built = True
            pass
        else:
            is_built = True
            self.generate_connections()

        return is_built