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
    
Playground_model_4 = torch.load('Models/Playground_model_4.pth')
Playground_model_10 = torch.load('Models/Playground_model_10.pth')

Parking_lot_model_4 = torch.load('Models/Parking_lot_model_4.pth')
Parking_lot_model_10 = torch.load('Models/Parking_lot_model_10.pth')

Office_model_2 = torch.load('Models/Office_model_2.pth')
Office_model_4 = torch.load('Models/Office_model_4.pth')
Office_model_10 = torch.load('Models/Office_model_10.pth')

Playground_test_df = pd.read_csv('Test_set/Playground_data_200_test_set.csv')
Playground_test_x_4 = Playground_test_df.iloc[:, :4].values
Playground_test_x_10 = Playground_test_df.iloc[:, :-1].values
Playground_test_y = Playground_test_df.iloc[:, -1].values

Parking_lot_test_df = pd.read_csv('Test_set/Parking_lot_data_200_test_set.csv')
Parking_lot_test_x_4 = Parking_lot_test_df.iloc[:, :4].values
Parking_lot_test_x_10 = Parking_lot_test_df.iloc[:, :-1].values
Parking_lot_test_y = Parking_lot_test_df.iloc[:, -1].values

Office_test_df = pd.read_csv('Test_set/Office_data_200_test_set.csv')
Office_test_x_2 = Office_test_df.iloc[:, :2].values
Office_test_x_4 = Office_test_df.iloc[:, :4].values
Office_test_x_10 = Office_test_df.iloc[:, :-1].values
Office_test_y = Office_test_df.iloc[:, -1].values

def model_predict(model, test_x):
    model.eval()
    X_test_tensor = torch.FloatTensor(test_x)
    test_outputs = model(X_test_tensor)
    return test_outputs.reshape(110,)

Playground_outputs_4 = model_predict(Playground_model_4, Playground_test_x_4)
Playground_outputs_10 = model_predict(Playground_model_10, Playground_test_x_10)

plt.figure()
plt.violinplot(Playground_outputs_4.detach().numpy().reshape(11,10).T)
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

Office_outputs_2 = model_predict(Office_model_2, Office_test_x_2)
Office_outputs_4 = model_predict(Office_model_4, Office_test_x_4)
Office_outputs_10 = model_predict(Office_model_10, Office_test_x_10)

rect_with_GMM = plt.Rectangle((0, 0), 1, 1, facecolor='skyblue')
rect_without_GMM = plt.Rectangle((0, 0), 1, 1, facecolor='coral')
error_bond = plt.Line2D((0, 1), (0, 1), color='y',linestyle='--')
groundtruth = plt.Line2D((0, 1), (0, 1), color='black')

plt.figure()
plt.violinplot(Office_outputs_4.detach().numpy().reshape(11,10).T)
plt.violinplot(Office_outputs_10.detach().numpy().reshape(11,10).T)
plt.plot([i for i in range(1,12)], [i for i in range(1,12)], c = 'black')
plt.plot([i for i in range(1,12)], [i+1 for i in range(1,12)], linestyle='--', c = 'y')
plt.plot([i for i in range(1,12)], [i-1 for i in range(1,12)], linestyle='--', c = 'y')
plt.grid()
plt.legend()
plt.xlabel('Distance (m)', fontdict={'weight': 'normal', 'size': 15})
plt.ylabel('Prediction Distance (m)', fontdict={'weight': 'normal', 'size': 15})
plt.xlim(0,12)
plt.ylim(0,12)
plt.legend([rect_with_GMM, rect_without_GMM,error_bond, groundtruth], ['ML-based RTT+RSSI\n Ranging with GMM', 'ML-based RTT+RSSI\n Ranging', '+/-1 m Error_boundray', 'True Distance'])
plt.rcParams.update({'font.size': 10})
plt.savefig('Figures/Office_results.svg',dpi=1000,format='svg')
plt.show()


Parking_lot_outputs_4 = model_predict(Parking_lot_model_4, Parking_lot_test_x_4)
Parking_lot_outputs_10 = model_predict(Parking_lot_model_10, Parking_lot_test_x_10)

plt.figure()
plt.violinplot(Parking_lot_outputs_4.detach().numpy().reshape(11,10).T)
plt.violinplot(Parking_lot_outputs_10.detach().numpy().reshape(11,10).T)
plt.plot([i for i in range(1,12)], [i for i in range(1,12)], c = 'black')
plt.plot([i for i in range(1,12)], [i+1 for i in range(1,12)], linestyle='--', c = 'y')
plt.plot([i for i in range(1,12)], [i-1 for i in range(1,12)], linestyle='--', c = 'y')
plt.grid()
plt.legend()
plt.xlabel('Distance (m)')
plt.ylabel('Prediction Distance (m)')
plt.xlim(0,12)
plt.ylim(0,12)
plt.savefig('Figures/Parking_lot_results.png')
plt.show()

def draw_cdf(outputs_4, outputs_10, test_y, data_name):
    test_y = torch.FloatTensor(test_y)
    error_4 = np.abs((test_y - outputs_4).detach().numpy())
    error_10 = np.abs((test_y - outputs_10).detach().numpy())
    error_df = pd.DataFrame({'{}_without_GMM'.format(data_name): error_4, '{}_with_GMM'.format(data_name): error_10})
    return error_df

def cal_error(outputs,test_y):
    test_y = torch.FloatTensor(test_y)
    error = np.abs((test_y - outputs).detach().numpy())
    return np.mean(error)

Playground_df = draw_cdf(Playground_outputs_4, Playground_outputs_10, Playground_test_y, 'Playground')
Office_df = draw_cdf(Office_outputs_4, Office_outputs_10, Office_test_y, 'Office')
Parking_lot_df = draw_cdf(Parking_lot_outputs_4, Parking_lot_outputs_10, Parking_lot_test_y, 'Parking_lot')

result = Playground_df.join(Office_df)
result = result.join(Parking_lot_df)
plt.figure()
sns.ecdfplot(data=result, legend=True)
plt.ylabel('Propotion')
plt.xlabel('Estimation Error(m)')
plt.grid()
plt.savefig('Figures/All_data_CDF.svg',dpi=1000,format='svg')
plt.show()

Playground_error_4 = cal_error(Playground_outputs_4, Playground_test_y)
Playground_error_10 = cal_error(Playground_outputs_10, Playground_test_y)

Parking_lot_error_4 = cal_error(Parking_lot_outputs_4, Parking_lot_test_y)
Parking_lot_error_10 = cal_error(Parking_lot_outputs_10, Parking_lot_test_y)

Office_error_2 = cal_error(Office_outputs_2, Office_test_y)
Office_error_4 = cal_error(Office_outputs_4, Office_test_y)
Office_error_10 = cal_error(Office_outputs_10, Office_test_y)

print('Playground error 4:', Playground_error_4)
print('Playground error 10:', Playground_error_10)

print('Parking_lot error 4:', Parking_lot_error_4)
print('Parking_lot error 10:', Parking_lot_error_10)

#print(np.mean([(x-y)**2 for x,y in zip(Office_outputs_4.detach().numpy(), Office_test_y)])**(0.5))

print('Office error 4:', Office_error_4)
print('Office error 10:', Office_error_10)

plt.figure()
plt.violinplot([Playground_df['Playground_without_GMM'], Playground_df['Playground_with_GMM'],
             Parking_lot_df['Parking_lot_without_GMM'], Parking_lot_df['Parking_lot_with_GMM'],
             Office_df['Office_without_GMM'], Office_df['Office_with_GMM']], showextrema=True, showmedians= True)
plt.show()

Office_error_2 = np.abs(Office_outputs_2.detach().numpy() - Office_test_y)
Office_error_4 = np.abs(Office_outputs_4.detach().numpy() - Office_test_y)
Office_error_10 = np.abs(Office_outputs_10.detach().numpy() - Office_test_y)

Office_error_1 = np.array([(i-1.5)*299792458/(16000000*2) for i in Office_test_df.iloc[:, :1].values]).reshape(110,) - Office_test_y
Office_error_1 = np.abs(Office_error_1)
plt.figure()
plt.boxplot([Office_error_10, Office_error_4, Office_error_2, Office_error_1], showfliers=False)
labels = ['ML-based \nRTT+RSSI\n Ranging with GMM', 'ML-based \nRTT+RSSI\n Ranging', 'ML-based RTT\n Ranging ', 'RTT-Ranging']
plt.xticks([1, 2, 3, 4,], labels, fontdict={'weight': 'normal', 'size': 8})
plt.setp(plt.gca().get_xticklabels(), multialignment='center')
plt.ylabel('Prediction Error(m)', fontdict={'weight': 'normal', 'size': 15})
plt.grid()
plt.savefig('Figures/Different_ranging_methods.svg',dpi=300,format='svg')
plt.show()



# 创建主图
ax1 = plt.subplot(1, 2, 1)  # 左侧图
ax1.boxplot([Office_error_10, Office_error_4, Office_error_2, Office_error_1], showfliers=False)
labels = ['ML-based \nRTT+RSSI\n Ranging with GMM', 'ML-based \nRTT+RSSI\n Ranging', 'ML-based RTT\n Ranging ', 'RTT-Ranging']
plt.xticks([1, 2, 3, 4], labels, fontdict={'weight': 'normal', 'size': 8})
plt.setp(plt.gca().get_xticklabels(), multialignment='center')
plt.ylabel('Prediction Error (m)', fontdict={'weight': 'normal', 'size': 15})
plt.title('Zoomed In on First Three Methods')
plt.grid()

tx0 = 0.7
tx1 = 3.5
ty0 = -0.5
ty1 = 3
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax1.plot(sx,sy,"purple")

# 创建第二个图（右侧图），包含最后一个箱线图
ax2 = plt.subplot(1, 2, 2)  # 右侧图
ax2.boxplot([Office_error_10, Office_error_4, Office_error_2], showfliers=False)
labels = ['ML-based \nRTT+RSSI\n Ranging with GMM', 'ML-based \nRTT+RSSI\n Ranging', 'ML-based RTT\n Ranging ']
plt.xticks([1, 2, 3], labels, fontdict={'weight': 'normal', 'size': 8})
plt.grid()

xy = (3,3)
xy2 = (0,3)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=ax2,axesB=ax1)
ax2.add_artist(con)

xy = (4.45,-0.09)
xy2 = (4.02,-0.018)
con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",
        axesA=ax2,axesB=ax1)
ax2.add_artist(con)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('different_ranging_methods_zoomed.svg', dpi=300, format='svg')
plt.show()