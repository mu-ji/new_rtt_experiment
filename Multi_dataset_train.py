import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# 读取训练集和测试集
train_df_1 = pd.read_csv('Train_set/Playground_data_200_train_set.csv')  # 替换为你的训练集文件名
train_df_2 = pd.read_csv('Train_set/Office_data_200_train_set.csv')  # 替换为你的训练集文件名
train_df_3 = pd.read_csv('Train_set/Parking_lot_data_200_train_set.csv')
train_df = pd.concat([train_df_1, train_df_2, train_df_3])
Playground_test_df = pd.read_csv('Test_set/Playground_data_200_test_set.csv')    # 替换为你的测试集文件名
Parking_lot_test_df = pd.read_csv('Test_set/Parking_lot_data_200_test_set.csv')
Office_test_df = pd.read_csv('Test_set/Office_data_200_test_set.csv')
# 假设前10列是特征，最后一列是标签
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

Playground_X_test = Playground_test_df.iloc[:, :-1].values
Playground_y_test = Playground_test_df.iloc[:, -1].values

Parking_lot_X_test = Parking_lot_test_df.iloc[:, :-1].values
Parking_lot_y_test = Parking_lot_test_df.iloc[:, -1].values

Office_X_test = Office_test_df.iloc[:, :-1].values
Office_y_test = Office_test_df.iloc[:, -1].values

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)  # 回归问题使用 FloatTensor
Playground_X_test_tensor = torch.FloatTensor(Playground_X_test)
Playground_y_test_tensor = torch.FloatTensor(Playground_y_test)

Parking_lot_X_test_tensor = torch.FloatTensor(Parking_lot_X_test)
Parking_lot_y_test_tensor = torch.FloatTensor(Parking_lot_y_test)

Office_X_test_tensor = torch.FloatTensor(Office_X_test)
Office_y_test_tensor = torch.FloatTensor(Office_y_test)

# 创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

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

# 超参数
input_size = X_train.shape[1]  # 特征数量
hidden_size = 256                 # 隐藏层神经元数量
num_epochs = 300
learning_rate = 0.001

# 实例化模型、损失函数和优化器
model = MLP(input_size, hidden_size)
criterion = nn.MSELoss()  # 回归问题使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)  # 去掉多余的维度
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 训练完成后，在测试集上评估模型
with torch.no_grad():
    model.eval()
    Playground_test_outputs = model(Playground_X_test_tensor)
    Playground_test_loss = criterion(Playground_test_outputs.squeeze(), Playground_y_test_tensor)  # 计算测试集损失
    Parking_lot_test_outputs = model(Parking_lot_X_test_tensor)
    Parking_lot_test_loss = criterion(Parking_lot_test_outputs.squeeze(), Parking_lot_y_test_tensor)  # 计算测试集损失
    Office_test_outputs = model(Office_X_test_tensor)
    Office_test_loss = criterion(Office_test_outputs.squeeze(), Office_y_test_tensor)  # 计算测试集损失
    print(f'Platground Test Loss: {Playground_test_loss.item():.4f}')
    print(f'Parking_lot Test Loss: {Parking_lot_test_loss.item():.4f}')
    print(f'Office Test Loss: {Office_test_loss.item():.4f}')



Playground_error = np.abs(Playground_y_test_tensor - Playground_test_outputs.reshape(110,))
Parking_lot_error = np.abs(Parking_lot_y_test_tensor - Parking_lot_test_outputs.reshape(110,))
Office_error = np.abs(Office_y_test_tensor - Office_test_outputs.reshape(110,))


plt.figure()
plt.violinplot([Playground_error, Parking_lot_error, Office_error], showmeans=True)
plt.xlabel('Value')
plt.ylabel('CDF')
plt.grid()
plt.show()

commend = input('Save model or not? (y/n)')
if commend == 'y':
    torch.save(model, 'Models/Multi_model_10.pth') #10 is the number of features
else:
    print('not save')