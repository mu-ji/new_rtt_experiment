import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# 读取训练集和测试集
train_df_1 = pd.read_csv('special_test/Playground_data_200_train_set.csv')  # 替换为你的训练集文件名
train_df_2 = pd.read_csv('special_test/Office_data_200_train_set.csv')  # 替换为你的训练集文件名
train_df_3 = pd.read_csv('special_test/Parking_lot_data_200_train_set.csv')
train_df = pd.concat([train_df_1, train_df_2, train_df_3])
test_df = pd.read_csv('special_test/Office_data_200_test_set.csv')    # 替换为你的测试集文件名

# 假设前10列是特征，最后一列是标签
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train)

y_train_tensor = torch.FloatTensor(y_train)  # 回归问题使用 FloatTensor
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# 创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.tanh = nn.ELU()
        self.fc2 = nn.Linear(512, 128)
        self.tanh = nn.ELU()
        self.fc3 = nn.Linear(128, 64)
        self.tanh = nn.ELU()
        self.fc4 = nn.Linear(64, 32)
        self.tanh = nn.ELU()
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
hidden_size = 128                 # 隐藏层神经元数量
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
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs.squeeze(), y_test_tensor)  # 计算测试集损失
    print(f'Test Loss: {test_loss.item():.4f}')

plt.figure()
plt.scatter(y_test_tensor, test_outputs)
plt.plot([i for i in range(1,12)], [i for i in range(1,12)], c = 'black')
plt.plot([i for i in range(1,12)], [i+1 for i in range(1,12)], linestyle='--', c = 'y')
plt.plot([i for i in range(1,12)], [i-1 for i in range(1,12)], linestyle='--', c = 'y')
plt.show()

error = np.abs(y_test_tensor - test_outputs.reshape(110,))
sorted_data = np.sort(error)
cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
error = pd.DataFrame({'Playground': error})

plt.figure()
sns.ecdfplot(data=error, legend=True)
plt.title('Cumulative Distribution Function (CDF)')
plt.xlabel('Value')
plt.ylabel('CDF')
plt.grid()
plt.show()

commend = input('Save model or not? (y/n)')
if commend == 'y':
    torch.save(model, 'Models/Playground_model_10.pth') #10 is the number of features
else:
    print('not save')