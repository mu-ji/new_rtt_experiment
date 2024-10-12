import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import random
import pandas as pd

packet_number = 30000
folder_name = 'test_data'
distance_list = [0,1]
sample_length = 200
sample_number_per_distance = 20

def generate_samples(start_index, rtt_data, rssi_data, sample_length):
    rtt_data = rtt_data[start_index:start_index+sample_length]
    rssi_data = rssi_data[start_index:start_index+sample_length]

    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(rtt_data.reshape(-1,1))

    weights = gmm.weights_
    covariances = gmm.covariances_
    means = gmm.means_

    return np.array([np.mean(rtt_data), np.var(rtt_data), np.mean(rssi_data), np.var(rssi_data), means[0][0], means[1][0], covariances[0][0][0], covariances[1][0][0], weights[0], weights[1]])

def update_train_set(train_set, sample, label):
    train_set['rtt_mean'].append(sample[0])
    train_set['rtt_var'].append(sample[1])
    train_set['rssi_mean'].append(sample[2])
    train_set['rssi_var'].append(sample[3])
    train_set['gmm_mean_1'].append(sample[4])
    train_set['gmm_mean_2'].append(sample[5])
    train_set['gmm_var_1'].append(sample[6])
    train_set['gmm_var_2'].append(sample[7])
    train_set['gmm_weight_1'].append(sample[8])
    train_set['gmm_weight_2'].append(sample[9])
    train_set['label'].append(label)
    return train_set

def generate_train_set(folder_name, distance_list, sample_length, sample_number_per_distance):
    train_set = {
        'rtt_mean': [],
        'rtt_var': [],
        'rssi_mean': [],
        'rssi_var': [],
        'gmm_mean_1': [],
        'gmm_mean_2': [],
        'gmm_var_1': [],
        'gmm_var_2': [],
        'gmm_weight_1': [],
        'gmm_weight_2': [],
        'label': []
    }
    for distance in distance_list:
        distance_data = np.load('{}/{}_distance_{}.npz'.format(folder_name, folder_name, distance))
        distance_rtt_data = distance_data['rtt_data']
        distance_rtt_data -= 8070
        for k in range(sample_number_per_distance):
            start_index = random.randint(20000, 30000-sample_length)
            sample = generate_samples(start_index, distance_rtt_data, data['rssi_data'], sample_length)
            train_set = update_train_set(train_set, sample, distance)
    return train_set

data = np.load('test_data/test_data_distance_0.npz')
sample = generate_samples(2, data['rtt_data'], data['rssi_data'], 20)
print(sample)

train_set = generate_train_set(folder_name, distance_list, sample_length, sample_number_per_distance)
print(train_set)
train_set = pd.DataFrame(train_set)
train_set.to_csv('train_set/{}_{}_train_set.csv'.format(folder_name, sample_length), index=False)
print(train_set)