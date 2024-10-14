import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


test_df = pd.read_csv('Test_set/Office_data_200_test_set.csv')    # 替换为你的测试集文件名

rtt_mean = test_df.iloc[:, 0]
y_test = test_df.iloc[:, -1]
estimation = [(i-1.1)*299792458/(16000000*2) for i in rtt_mean]

plt.figure()
plt.scatter(y_test, estimation)
plt.plot([i for i in range(1,12)], [i for i in range(1,12)], c = 'black')
plt.plot([i for i in range(1,12)], [i+1 for i in range(1,12)], linestyle='--', c = 'y')
plt.plot([i for i in range(1,12)], [i-1 for i in range(1,12)], linestyle='--', c = 'y')
plt.show()