import numpy as np
import math


def min_natural_number(array):
    max_num = max(array)
    array = np.where(array < 0, max_num, array)

    return min(array)


def get_incoming_connections(connections, index):
    incoming_connections = np.swapaxes(connections, 0, 1)[index]
    y = incoming_connections[:, :, 0].flatten()
    w = incoming_connections[:, :, 1].flatten()

    return y, w


def get_outgoing_connections(connections, index):
    outgoing_connections = connections[index]
    y = outgoing_connections[:, :, 0].flatten()
    w = outgoing_connections[:, :, 1].flatten()

    return y, w


def epsilon(tau, t):
    return t/tau * np.exp(1-(t/tau))


def y(tau, t, t_i, d):
    return epsilon(tau, t - t_i - d)


def mask(current_time, prev_firing_times, n_terminal, delay):
    t_mask = []

    for firing_time in prev_firing_times:
        for i_terminal in range(n_terminal):
            t_mask.append(firing_time + (delay * i_terminal + 1))

    t_mask = np.array(t_mask) <= current_time

    return t_mask


def flatten(array):
    t_array = np.array(array)

    return t_array.flatten()


def transform_inverse_firing_time(a_value, max_firing_time, threshold):
    norm_value = (np.array(a_value) - np.min(a_value)) / np.max(a_value) - np.min(a_value)      # min-max scaling

    inversed = np.round((1 - norm_value) * max_firing_time) + 1
    inversed = np.where(inversed >= threshold, -1, inversed)

    return inversed


def norm_pdf(x, mu, sigma):
    PI = 3.141592

    return np.exp(-((x - mu)**2) / (2 * (sigma**2))) / (math.sqrt(2 * PI) * sigma)


def Y_encoder(src_Y, size, desired_firing_time, gap):
    dst_Y = np.full(size, desired_firing_time + gap)
    dst_Y[src_Y] = desired_firing_time

    return dst_Y


def GRF_encoder(record, r_value, n_neurons, max_firing_time):
    min = r_value[0]
    max = r_value[1]
    encoded_neurons = []
    gamma = 3
    threshold = max_firing_time

    for input_value in record:
        for i in range(n_neurons):
            mu = min + ((2 * (i + 1) - 3) / 2) * ((max - min) / (n_neurons - 2))
            sigma = (1 / gamma) * ((max - min) / (n_neurons - 2))
            t_neuron = norm_pdf(input_value, mu, sigma)

            encoded_neurons.append(t_neuron)

    encoded_neurons = np.array(encoded_neurons)
    encoded_neurons = transform_inverse_firing_time(encoded_neurons, max_firing_time, threshold)

    return encoded_neurons


#y = utils.get_y(utils.flatten(y), current_time, prev_layer.neurons, self.delay, self.tau,self.n_terminals)  # conversion y into 1-dimensional vector.


#y, current_time, neuron, self.delay, self.tau, self.n_terminals
def get_y(array, t, array_t_i, delay, tau, n_terminals):
    n_neurons = len(array) // n_terminals

    for i in range(n_neurons * n_terminals):
        t_i = array_t_i[i // n_terminals]
        d = delay * ((i % n_terminals) + 1)

        if t <= t_i or t_i <= 0 or t <= t_i + d:
            array[i] = 0
            pass
        else:
            # print('...')
            array[i] = y(tau, t, t_i, d)

    return array


def update_connections(src_connections, dst_y, dst_w, i_neuron):
    src_shape = src_connections.shape
    updated = np.swapaxes(src_connections, 0, 1)

    temp_shape = updated.shape

    shape = (temp_shape[1], temp_shape[2], 1)

    updated_y = np.reshape(dst_y, shape)
    updated_w = np.reshape(dst_w, shape)
    updated_y_w = np.concatenate((updated_y, updated_w), axis=2)
    updated[i_neuron] = updated_y_w

    src_connections = np.reshape(updated, src_shape)

    # print("*** y and w were updated! ***")


def mse_loss(list_t_a, list_t_d):
    loss = 0

    for t_a, t_d in zip(list_t_a, list_t_d):
        loss = (t_a - t_d)**2
    loss = loss / 2

    return loss


def diff_y_t(t, t_i, d, tau):
    # partial derivative respect to t --> round y round t
    if t - t_i - d <= 0:
        return 0
    return np.exp(1 - (t - t_i - d) / tau) / tau - (t - t_i - d) * np.exp(1 - (t - t_i - d) / tau) / tau**2


def get_delta(i_neuron, l_connections, tau, d, n_terminals, is_output_layer, t_i, t_d=None, t_a=None, t_h=None, t_j=None,  prev_delta=None):
    delta = None
    # if t_a == -1:
    #     t_a = 40

    if is_output_layer == True:
        # delta for output layer
        y, w = get_incoming_connections(l_connections[0], i_neuron)

        for i in range(len(y)):
            delay = d * (i % n_terminals + 1)           # delays for each terminals
            y[i] = diff_y_t(t_a, t_i[i // n_terminals], delay, tau)       # round y

        if np.sum(w * y) == 0:
            return 0
        else:
            delta = (t_d - t_a) / np.sum(w * y)
    else:
        # delta for generalized cases (hidden layer)
        next_connections = l_connections[0]     # i x j
        curr_connections = l_connections[1]     # h x i

        y_i, w_ij = get_outgoing_connections(next_connections, i_neuron)        # connections into neuron i from neuron h
        y_h, w_hj = get_incoming_connections(curr_connections, i_neuron)        # connections from neuron i into neuron j

        for i in range(len(y_i)):
            delay = d * (i % n_terminals + 1)           # delays for each terminals
            y_i[i] = diff_y_t(t_j[i // n_terminals], t_i, delay, tau)       # round y_i

        for i in range(len(y_h)):
            delay = d * (i % n_terminals + 1)           # delays for each terminals
            y_h[i] = diff_y_t(t_i, t_h[i // n_terminals], delay, tau)       # round y_h

        # make mask for prev_delta * w_ij * y_i.
        prev_delta_mask = []
        for i in range(n_terminals * len(t_j)):
            prev_delta_mask.append(prev_delta[i // n_terminals])
        prev_delta_mask = np.array(prev_delta_mask)

        if np.sum(w_hj * y_h) == 0:
            return 0
        else:
            delta = np.sum(prev_delta_mask * w_ij * y_i) / np.sum(w_hj * y_h)

    return delta


def init_layers(layer_list, value=-1):
    for layer in layer_list:
        layer.neurons = np.full(layer.neurons.shape, -1)

        if layer_list.index(layer) == 0:
            # if first layer then there is not needed to init the connections.
            pass
        else:
            # init y to 0.
            connections = np.swapaxes(layer.connections, 0, 1)
            connections[:, :, :, 0] = np.zeros(connections[:, :, :, 0].shape)
            connections = np.swapaxes(layer.connections, 0, 1)

            # y = np.zeros(y.shape)
            #
            # temp_shape = connections.shape
            # shape = (temp_shape[1], temp_shape[2], 1)
            #
            # updated_y = np.reshape(y, temp_shape)
            # updated_w = np.reshape(w, temp_shape)
            # updated_y_w = np.concatenate((updated_y, updated_w), axis=2)
            #
            # layer.connections = updated_y_w


def convert_not_fired(neurons, value):
    for i_neuron in range(len(neurons)):
        if neurons[i_neuron] < 0:
            # not fired
            neurons[i_neuron] = value

    return neurons

