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
    
Multi_model_10 = torch.load('Models/Multi_model_10.pth')

Playground_test_df = pd.read_csv('Test_set/Playground_data_200_test_set.csv')
Playground_test_x_10 = Playground_test_df.iloc[:, :-1].values
Playground_test_y = Playground_test_df.iloc[:, -1].values

Parking_lot_test_df = pd.read_csv('Test_set/Parking_lot_data_200_test_set.csv')
Parking_lot_test_x_10 = Parking_lot_test_df.iloc[:, :-1].values
Parking_lot_test_y = Parking_lot_test_df.iloc[:, -1].values

Office_test_df = pd.read_csv('Test_set/Office_data_200_test_set.csv')
Office_test_x_10 = Office_test_df.iloc[:, :-1].values
Office_test_y = Office_test_df.iloc[:, -1].values

def model_predict(model, test_x):
    model.eval()
    X_test_tensor = torch.FloatTensor(test_x)
    test_outputs = model(X_test_tensor)
    return test_outputs.reshape(110,)

Playground_outputs_10 = model_predict(Multi_model_10, Playground_test_x_10)

plt.figure()
plt.scatter(Playground_test_y, Playground_outputs_10.detach().numpy(), label='Multi_train_Playground_test')
plt.plot([i for i in range(1,12)], [i for i in range(1,12)], c = 'black')
plt.plot([i for i in range(1,12)], [i+1 for i in range(1,12)], linestyle='--', c = 'y')
plt.plot([i for i in range(1,12)], [i-1 for i in range(1,12)], linestyle='--', c = 'y')
plt.grid()
plt.legend()
plt.xlabel('Distance (m)')
plt.ylabel('Prediction Distance (m)')
plt.xlim(0,12)
plt.ylim(0,12)
plt.savefig('Figures/Multi_train_Playground_test.png')
plt.show()

Office_outputs_10 = model_predict(Multi_model_10, Office_test_x_10)

plt.figure()
plt.scatter(Office_test_y, Office_outputs_10.detach().numpy(), label='Multi_train_Office_test')
plt.plot([i for i in range(1,12)], [i for i in range(1,12)], c = 'black')
plt.plot([i for i in range(1,12)], [i+1 for i in range(1,12)], linestyle='--', c = 'y')
plt.plot([i for i in range(1,12)], [i-1 for i in range(1,12)], linestyle='--', c = 'y')
plt.grid()
plt.legend()
plt.xlabel('Distance (m)')
plt.ylabel('Prediction Distance (m)')
plt.xlim(0,12)
plt.ylim(0,12)
plt.savefig('Figures/Multi_train_Office_test.png')
plt.show()

Parking_lot_outputs_10 = model_predict(Multi_model_10, Parking_lot_test_x_10)

plt.figure()
plt.scatter(Parking_lot_test_y, Parking_lot_outputs_10.detach().numpy(), label='Multi_train_Parking_lot_test')
plt.plot([i for i in range(1,12)], [i for i in range(1,12)], c = 'black')
plt.plot([i for i in range(1,12)], [i+1 for i in range(1,12)], linestyle='--', c = 'y')
plt.plot([i for i in range(1,12)], [i-1 for i in range(1,12)], linestyle='--', c = 'y')
plt.grid()
plt.legend()
plt.xlabel('Distance (m)')
plt.ylabel('Prediction Distance (m)')
plt.xlim(0,12)
plt.ylim(0,12)
plt.savefig('Figures/Multi_train_Parking_lot_test.png')
plt.show()

def draw_cdf(outputs_10, test_y, data_name):
    test_y = torch.FloatTensor(test_y)
    error_10 = np.abs((test_y - outputs_10).detach().numpy())
    error_df = pd.DataFrame({'{}'.format(data_name): error_10})
    return error_df

def cal_error(outputs,test_y):
    test_y = torch.FloatTensor(test_y)
    error = np.abs((test_y - outputs).detach().numpy())
    return np.mean(error)

Playground_df = draw_cdf(Playground_outputs_10, Playground_test_y, 'Playground')
Office_df = draw_cdf(Office_outputs_10, Office_test_y, 'Office')
Parking_lot_df = draw_cdf(Parking_lot_outputs_10, Parking_lot_test_y, 'Parking_lot')

result = Playground_df.join(Office_df)
result = result.join(Parking_lot_df)
plt.figure()
sns.ecdfplot(data=Playground_df, legend=True, color = 'b', label = 'Playground')
sns.ecdfplot(data=Parking_lot_df, legend=True, palette='Oranges', label = 'Parking_lot')
sns.ecdfplot(data=Office_df, legend=True, palette = 'Greens', label = 'Office')
plt.ylabel('CDF', fontdict={'weight': 'normal', 'size': 12})

datasets = [Playground_df, Office_df, Parking_lot_df]
labels = ['Playground', 'Office', 'Parking_lot']
label_offsets = [0, 0.04, 0.08]  # Different vertical offsets for labels

for i, (df, label) in enumerate(zip(datasets, labels)):
    error_values = df[label].values
    error_80 = np.percentile(error_values, 80)
    
    # Draw a vertical line at the 80% error position
    plt.axvline(x=error_80, linestyle='--', color='gray', alpha=0.5)
    
    # Place the error value on the x-axis with staggered positioning
    y_offset = label_offsets[i]  # Use the corresponding offset
    plt.text(error_80, y_offset, f'{error_80:.2f}', 
             horizontalalignment='center', fontsize=9)
    
plt.xlabel('Prediction Error(m)', fontdict={'weight': 'normal', 'size': 12})
plt.grid()
plt.legend()
plt.savefig('Figures/Multi_train_all_data_CDF.svg',dpi=1000,format='svg')
plt.show()


Playground_error_10 = cal_error(Playground_outputs_10, Playground_test_y)
Parking_lot_error_10 = cal_error(Parking_lot_outputs_10, Parking_lot_test_y)
Office_error_10 = cal_error(Office_outputs_10, Office_test_y)


print('Playground error 10:', Playground_error_10)

print('Parking_lot error 10:', Parking_lot_error_10)

print('Office error 10:', Office_error_10)