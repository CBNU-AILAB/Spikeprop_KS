import numpy as np
import math


def min_natural_number(array):
    max_num = max(array)
    array = np.where(array < 0, max_num, array)

    return min(array)


def get_inner_connections(connections, index):
    inner_connections = np.swapaxes(connections, 0, 1)[index]
    y = inner_connections[:, :, 0]
    w = inner_connections[:, :, 1]

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
    dst_Y = []

    for y in src_Y:
        temp = np.full(size, desired_firing_time + gap)
        temp[y] = desired_firing_time

        dst_Y.append(temp)

    return dst_Y


def GRF_encoder(record, r_value, n_neurons, max_firing_time):
    min = r_value[0]
    max = r_value[1]
    encoded_neurons = []
    gamma = 1.5
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

#y, current_time, neuron, self.delay, self.tau, self.n_terminals
def get_y(array, t, array_t_i, delay, tau, n_terminals):
    n_neurons = len(array) // n_terminals

    for i in range(n_neurons * n_terminals):
        t_i = array_t_i[i // n_terminals]
        d = delay * ((i % n_terminals) + 1)
        if t <= t_i or t_i < 0 or t < t_i + d:
            pass
        else:
            print('...')
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

    print("*** y and w were updated! ***")


def mse_loss(t_a, t_d):
    return 1 / 2 * (np.sum((t_a - t_d)**2))


def diff_y_t(t, t_i, d, tau):
    # partial derivative respect to t --> round y round t
    return np.exp(1 - (t - t_i - d) / tau) / tau - (t - t_i - d) * np.exp(1 - (t - t_i - d) / tau) / tau**2


def get_delta(i_neuron, connections, t_d, t_a, t_i, tau, d, n_terminals, is_output_layer, prev_delta=None):
    delta = None
    if is_output_layer == True:
        # delta for output layer
        y, w = get_inner_connections(connections, i_neuron)

        for i in range(y):
            delay = d * (i % n_terminals + 1)           # delays for each terminals
            y[i] = diff_y_t(t_a, t_i[i // n_terminals], delay, tau)       # round y

        delta = (t_d - t_a) / np.sum(w * y)
    else:
        # delta for generalized cases (hidden layer)
        pass

    return delta