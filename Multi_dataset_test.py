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
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.elu = nn.ELU()
        self.fc3 = nn.Linear(hidden_size, 1)  # 输出层只有一个神经元

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
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
sns.ecdfplot(data=result, legend=True)
plt.ylabel('Propotion')
plt.xlabel('Estimation Error(m)')
plt.grid()
plt.savefig('Figures/Multi_train_all_data_CDF.png')
plt.show()


Playground_error_10 = cal_error(Playground_outputs_10, Playground_test_y)
Parking_lot_error_10 = cal_error(Parking_lot_outputs_10, Parking_lot_test_y)
Office_error_10 = cal_error(Office_outputs_10, Office_test_y)


print('Playground error 10:', Playground_error_10)

print('Parking_lot error 10:', Parking_lot_error_10)

print('Office error 10:', Office_error_10)