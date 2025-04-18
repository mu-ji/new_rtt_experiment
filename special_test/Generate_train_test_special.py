import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import random
import pandas as pd

packet_number = 30000
folder_name = 'Playground_data'
distance_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
sample_length = 200
sample_number_per_distance = 40
sample_number_per_distance_test = 10

def generate_samples(start_index, rtt_data, rssi_data, sample_length):
    rtt_data = rtt_data[start_index:start_index+sample_length]
    rssi_data = rssi_data[start_index:start_index+sample_length]
    # add gmm selection, only return a distribution after GMM
    return np.array([rtt_data, rssi_data]).reshape(-1)

def update_train_set(train_set, sample, label):
    train_set.append(np.append(sample,label))

    return train_set

def generate_train_set(folder_name, distance_list, sample_length, sample_number_per_distance):
    train_set = []
    for distance in distance_list:
        distance_data = np.load('{}/{}_distance_{}.npz'.format(folder_name, folder_name, distance))
        distance_rtt_data = distance_data['rtt_data']
        distance_rtt_data -= 8073
        for k in range(sample_number_per_distance):
            start_index = random.randint(10000, 30000-sample_length)
            sample = generate_samples(start_index, distance_rtt_data, distance_data['rssi_data'], sample_length)
            train_set = update_train_set(train_set, sample, distance)
    return train_set

def generate_test_set(folder_name, distance_list, sample_length, sample_number_per_distance_test):
    test_set = []
    for distance in distance_list:
        distance_data = np.load('{}/{}_distance_{}.npz'.format(folder_name, folder_name, distance))
        distance_rtt_data = distance_data['rtt_data']
        distance_rtt_data -= 8073
        for k in range(sample_number_per_distance_test):
            start_index = random.randint(0, 10000-sample_length)
            sample = generate_samples(start_index, distance_rtt_data, distance_data['rssi_data'], sample_length)
            test_set = update_train_set(test_set, sample, distance)
    return test_set

#data = np.load('test_data/test_data_distance_0.npz')
#sample = generate_samples(2, data['rtt_data'], data['rssi_data'], 20)
#print(sample)

train_set = generate_train_set(folder_name, distance_list, sample_length, sample_number_per_distance)
test_set = generate_test_set(folder_name, distance_list, sample_length, sample_number_per_distance_test)
print(train_set)
train_set = pd.DataFrame(train_set)
test_set = pd.DataFrame(test_set)
train_set.to_csv('special_test/{}_{}_train_set.csv'.format(folder_name, sample_length), index=False)
test_set.to_csv('special_test/{}_{}_test_set.csv'.format(folder_name, sample_length), index=False)
print(train_set)