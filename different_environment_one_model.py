import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import ConnectionPatch

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
    

Playground_model_10 = torch.load('Models/Playground_model_10.pth')

Parking_lot_model_10 = torch.load('Models/Parking_lot_model_10.pth')

Office_model_10 = torch.load('Models/Office_model_10.pth')

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

Playground_outputs_10 = model_predict(Playground_model_10, Parking_lot_test_x_10)

plt.figure()
plt.violinplot(Playground_outputs_10.detach().numpy().reshape(11,10).T)
plt.plot([i for i in range(1,12)], [i for i in range(1,12)], c = 'black')
plt.plot([i for i in range(1,12)], [i+1 for i in range(1,12)], linestyle='--', c = 'y')
plt.plot([i for i in range(1,12)], [i-1 for i in range(1,12)], linestyle='--', c = 'y')
plt.grid()
plt.legend()
plt.xlabel('Distance (m)')
plt.ylabel('Prediction Distance (m)')
plt.xlim(0,12)
plt.ylim(0,12)
plt.savefig('Figures/Playground_results.png')
plt.show()

def draw_cdf(outputs_4, test_y, data_name):
    test_y = torch.FloatTensor(test_y)
    error_4 = np.abs((test_y - outputs_4).detach().numpy())
    error_df = pd.DataFrame({'{}'.format(data_name): error_4})
    return error_df

Playground_model_parking_lot = model_predict(Playground_model_10, Parking_lot_test_x_10)
Playground_model_office = model_predict(Playground_model_10, Office_test_x_10)

Office_model_parking_lot = model_predict(Office_model_10, Parking_lot_test_x_10)
Office_model_playground = model_predict(Office_model_10, Playground_test_x_10)

Parking_lot_model_playground = model_predict(Parking_lot_model_10, Playground_test_x_10)
Parking_lot_model_office = model_predict(Parking_lot_model_10, Office_test_x_10)


Playground_model_parking_lot_cdf = draw_cdf(Playground_outputs_10, Parking_lot_test_y, 'Playground_model_parking_lot_cdf')
Playground_model_office_cdf = draw_cdf(Playground_model_office, Office_test_y, 'Playground_model_office_cdf')

Office_model_parking_lot_cdf = draw_cdf(Office_model_parking_lot, Parking_lot_test_y, 'Office_model_parking_lot_cdf')
Office_model_playground_cdf = draw_cdf(Office_model_playground, Playground_test_y, 'Office_model_playground_cdf')

Parking_lot_model_playground_cdf = draw_cdf(Parking_lot_model_playground, Playground_test_y, 'Parking_lot_model_playground_cdf')
Parking_lot_model_office_cdf = draw_cdf(Parking_lot_model_office, Office_test_y, 'Parking_lot_model_office_cdf')

plt.figure(figsize=(8, 6))

sns.ecdfplot(data=Playground_model_parking_lot_cdf, color = 'b', label = 'Playground_model_parking_lot')
sns.ecdfplot(data=Playground_model_office_cdf, color = 'b', label = 'Playground_model_office', linestyle='--')

sns.ecdfplot(data=Office_model_parking_lot_cdf, palette='Oranges', label = 'Office_model_parking_lot')
sns.ecdfplot(data=Office_model_playground_cdf, palette='Oranges', label = 'Office_model_playground', linestyle='--')


sns.ecdfplot(data=Parking_lot_model_playground_cdf, palette = 'Greens', label = 'Parking_lot_model_playground')
sns.ecdfplot(data=Parking_lot_model_office_cdf, palette = 'Greens', label = 'Parking_lot_model_office', linestyle='--')

label_offsets = [0, 0.04, 0.08, 0.12, 0.16, 0.2]  # Different vertical offsets for labels

for i, (df, label) in enumerate(zip(
    [Playground_model_parking_lot_cdf, Playground_model_office_cdf, 
     Office_model_parking_lot_cdf, Office_model_playground_cdf,
     Parking_lot_model_playground_cdf, Parking_lot_model_office_cdf],
    ['Playground_model_parking_lot_cdf', 'Playground_model_office_cdf',
     'Office_model_parking_lot_cdf', 'Office_model_playground_cdf',
     'Parking_lot_model_playground_cdf', 'Parking_lot_model_office_cdf'])):
    
    error_values = df[label].values
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
plt.legend()
plt.savefig('Figures/Diff_model_environment.svg',dpi=1000,format='svg')
plt.show()