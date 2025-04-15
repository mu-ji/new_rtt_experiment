import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import random
import pandas as pd
from collections import Counter

packet_number = 30000
folder_name = 'Office_data'
distance_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
sample_length = 300
sample_number_per_distance = 80
sample_number_per_distance_test = 10

def generate_samples(start_index, rtt_data, sample_length):
    rtt_data = rtt_data[start_index:start_index+sample_length]
    elements_counts = Counter(rtt_data)


    return elements_counts

distance_data = np.load('{}/{}_distance_{}.npz'.format(folder_name, folder_name, distance_list[2]))
distance_rtt_data = distance_data['rtt_data']

print(generate_samples(26000,distance_rtt_data,200))