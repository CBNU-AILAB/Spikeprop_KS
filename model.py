import keras
import numpy as np
import losses
import utils
from queue import Queue


def get_index_terminal(prev_firing_time, current_time, delay):
    index = (current_time - prev_firing_time) // delay
    return index


class SP_Sequential:
    def __init__(self, n_terminals, delay, tau, theta):
        self.n_terminals = n_terminals
        self.delay = delay
        self.tau = tau
        self.theta = theta
        self.neurons = []
        self.connections = []    # list of connections between each neurons
        self.layer_list = []
        self.rl = 0.01
        self.loss_function = None


    def add(self, layer):
        # calculate input dimension on previous layer.
        if len(self.layer_list) != 0:
            layer.input_dim = self.layer_list[-1].output_dim

        # append layer into list of layers.
        layer.n_terminals = self.n_terminals
        self.layer_list.append(layer)


    def build_network(self):
        is_built = False

        for layer in self.layer_list:
            if layer.build() == True:
                self.neurons.append(layer.neurons)
                self.connections.append(layer.connections)
                is_built = True
            else:
                raise NotImplementedError

        return is_built


    def compile(self, loss='mse', optimizer='my_optimizer', lr='0.01'):
        self.lr = lr

        # look up the given loss from list of losses.
        loss_function = losses.find_loss(loss)
        if loss_function == None:
            raise NotImplementedError
        else:
            self.loss_function = loss

        # build added layers
        if self.build_network() == True:
            print("The network was built.")
        else:
            print("Network build Error!")

        return None


    def fit(self, input_data, output_data, epochs, batch_size=1):
        itr = 0
        while itr < 20:
            # do feed forward
            self.forward(itr, input_data)
            print(self.layer_list[2].neurons)
            # do backprop
            self.backward(itr, output_data)


            utils.init_layers(self.layer_list)  # init every neurons to -1.

            itr = itr + 1       # increase step


        return None


    def forward(self, itr, input_data):
        print("--step#%d feed forward--" % itr)
        is_input_layer = True
        prev_layer = None
        current_time = 1
        TIME_OVER = 200
        INTERVAL = 1
        is_done = False
        queue = Queue()

        while True:
            if current_time > TIME_OVER or is_done == True:
                break
            else:
                i_layer = 0
                for layer in self.layer_list:
                    if self.layer_list.index(layer) == 0:
                        # first layer --> just feed input data
                        self.layer_list[0].neurons = input_data
                        prev_layer = self.layer_list[0]
                        is_input_layer = False
                        # current_time = utils.min_natural_number(prev_layer.neurons) + INTERVAL
                    else:
                        # do calculation for all neurons in this layer.
                        i_neuron = 0        # index of neuron

                        for neuron in layer.neurons:
                            if neuron > 0:
                                # it's time over or current neuron is already fired.
                                i_neuron = i_neuron + 1
                                break
                            else:
                                # t_mask = utils.mask(current_time, prev_layer.neurons, self.n_terminals, self.delay)
                                y, w = utils.get_incoming_connections(layer.connections, i_neuron)
                                y = utils.get_y(utils.flatten(y), current_time, prev_layer.neurons, self.delay, self.tau, self.n_terminals)    # conversion y into 1-dimensional vector.
                                w = utils.flatten(w)    # conversion w into 1-dimensional vector.

                                masked_inner_connections = y * w
                                x = masked_inner_connections.sum()          # get its membrane potential.

                                if x >= self.theta:
                                    # membrane potential is crossed with the threshold theta.
                                    self.layer_list[i_layer].neurons[i_neuron] = current_time

                                # update y
                                utils.update_connections(self.layer_list[i_layer].connections, y, w, i_neuron)

                                i_neuron = i_neuron + 1         # increase the index of neuron.

                    i_layer = i_layer + 1                       # increase the index of layer.

            current_time += INTERVAL                            # increase the current time.


    def backward(self, itr, output_data):
        print("--step#%d backward--" % itr)

        error = 0

        # t_a = self.layer_list[-1].neurons
        # t_d = output_data[itr]
        #
        # if self.loss_function == 'mse':
        #     error = utils.mse_loss(t_a, t_d)

        prev_delta = []
        temp_prev_delta = []
        for layer in reversed(self.layer_list):
            i_current_layer = self.layer_list.index(layer)    # index of current layer

            if i_current_layer == 0:
                # current layer is the first layer(input layer).
                break

            prev_layer = self.layer_list[i_current_layer - 1]   # previous layer

            i_neuron = 0
            for neuron in layer.neurons:
                if i_current_layer == (len(self.layer_list) - 1):
                    # for output layer
                    delta = utils.get_delta(i_neuron=i_neuron,
                                            l_connections=[layer.connections],
                                            t_d=output_data[i_neuron],
                                            t_a=neuron,
                                            t_i=prev_layer.neurons,
                                            tau=self.tau,
                                            d=self.delay,
                                            n_terminals=self.n_terminals,
                                            is_output_layer=True,
                                            prev_delta=None)

                    y, w = utils.get_incoming_connections(layer.connections, i_neuron)
                    delta_w = -(self.lr * y * delta)
                    w = w + delta_w         # update weights
                    utils.update_connections(layer.connections, y, w, i_neuron)
                    temp_prev_delta.append(delta)
                    i_neuron = i_neuron + 1
                else:
                    # for hidden layer (generalied case)
                    if i_current_layer == len(self.layer_list):
                        # first layer --> end point of backwarding
                        break

                    next_layer = self.layer_list[i_current_layer + 1]       # layer J
                    delta = utils.get_delta(i_neuron=i_neuron,
                                            l_connections=[next_layer.connections, layer.connections],
                                            t_j=self.layer_list[i_current_layer + 1].neurons,
                                            t_i=neuron,
                                            t_h=self.layer_list[i_current_layer - 1].neurons,
                                            tau=self.tau,
                                            d=self.delay,
                                            n_terminals=self.n_terminals,
                                            is_output_layer=False,
                                            prev_delta=prev_delta)

                    y, w = utils.get_incoming_connections(layer.connections, i_neuron)
                    delta_w = self.lr * y * delta
                    w = w + delta_w         # update weights
                    utils.update_connections(layer.connections, y, w, i_neuron)
                    temp_prev_delta.append(delta)
                    i_neuron = i_neuron + 1

            prev_delta.clear()
            prev_delta = temp_prev_delta.copy()
            temp_prev_delta.clear()

        return None