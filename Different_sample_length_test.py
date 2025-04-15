import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(256, 128)
        self.tanh = nn.Tanh()
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()
        self.fc4 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()
        self.fc5 = nn.Linear(32, 1)  # 输出层只有一个神经元

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        return x
    
model_50 = torch.load('Models/Office_model_10_50.pth')
model_100 = torch.load('Models/Office_model_10_100.pth')
model_200 = torch.load('Models/Office_model_10.pth')
model_300 = torch.load('Models/Office_model_10_300.pth')

Office_test_50 = pd.read_csv('Test_set/Office_data_50_test_set.csv')
Office_test_x_50 = Office_test_50.iloc[:, :-1].values

Office_test_100 = pd.read_csv('Test_set/Office_data_100_test_set.csv')
Office_test_x_100 = Office_test_100.iloc[:, :-1].values

Office_test_200 = pd.read_csv('Test_set/Office_data_200_test_set.csv')
Office_test_x_200 = Office_test_200.iloc[:, :-1].values

Office_test_300 = pd.read_csv('Test_set/Office_data_300_test_set.csv')
Office_test_x_300 = Office_test_300.iloc[:, :-1].values

Office_test_y = Office_test_50.iloc[:, -1].values

def model_predict(model, test_x):
    model.eval()
    X_test_tensor = torch.FloatTensor(test_x)
    test_outputs = model(X_test_tensor)
    return test_outputs.reshape(110,)

Office_outputs_50 = model_predict(model_50, Office_test_x_50)
Office_outputs_100 = model_predict(model_100, Office_test_x_100)
Office_outputs_200 = model_predict(model_200, Office_test_x_200)
Office_outputs_300 = model_predict(model_300, Office_test_x_300)

def draw_cdf(outputs_10, test_y, data_name):
    test_y = torch.FloatTensor(test_y)
    error_10 = np.abs((test_y - outputs_10).detach().numpy())
    error_df = pd.DataFrame({'{}'.format(data_name): error_10})
    return error_df

def cal_error(outputs,test_y):
    test_y = torch.FloatTensor(test_y)
    error = np.abs((test_y - outputs).detach().numpy())
    return np.mean(error)

Office_50_df = draw_cdf(Office_outputs_50, Office_test_y, 'Number of packets = 50')
Office_100_df = draw_cdf(Office_outputs_100, Office_test_y, 'Number of packets = 100')
Office_200_df = draw_cdf(Office_outputs_200, Office_test_y, 'Number of packets = 200')
Office_300_df = draw_cdf(Office_outputs_300, Office_test_y, 'Number of packets = 300')

result = Office_50_df.join(Office_100_df)
result = result.join(Office_200_df)
result = result.join(Office_300_df)

plt.figure()
sns.ecdfplot(data=result, legend=True)

# Draw vertical lines and staggered labels on the x-axis for each dataset
labels = ['Number of packets = 50', 'Number of packets = 100', 
          'Number of packets = 200', 'Number of packets = 300']

label_offsets = [0, 0.04, 0.08, 0.12]  # Different vertical offsets for labels

for i, label in enumerate(labels):
    error_values = result[label].values
    error_80 = np.percentile(error_values, 80)
    
    # Draw a vertical line at the 80% error position
    plt.axvline(x=error_80, linestyle='--', color='gray', alpha=0.5)
    
    # Place the error value on the x-axis with staggered positioning
    y_offset = label_offsets[i]  # Use the corresponding offset
    plt.text(error_80, y_offset, f'{error_80:.2f}', 
             horizontalalignment='center', fontsize=9)
    
plt.ylabel('CDF', fontdict={'weight': 'normal', 'size': 12})
plt.xlabel('Prediction Error(m)', fontdict={'weight': 'normal', 'size': 12})
plt.grid()
plt.savefig('Figures/Different_samplelength_office_CDF.svg',dpi=1000,format='svg')
plt.show()