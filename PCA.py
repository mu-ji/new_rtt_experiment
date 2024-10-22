import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv('Train_set/Parking_lot_data_200_train_set.csv')


# 假设前10列是特征，最后一列是标签
X_train = train_df.iloc[:, :-8].values
y_train = train_df.iloc[:, -1].values

# 数据中心化
data_meaned = X_train - np.mean(X_train, axis=0)

# 计算协方差矩阵
covariance_matrix = np.cov(data_meaned, rowvar=False)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# 将特征值按降序排序
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# 选择前k个特征向量（例如选择前2个主成分）
k = 3
eigenvector_subset = eigenvectors[:, :k]

# 将数据投影到新的特征空间
data_reduced = np.dot(data_meaned, eigenvector_subset)

# 可视化降维后的数据
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_reduced[:, 0], data_reduced[:, 1], data_reduced[:, 2], c=y_train, cmap='viridis', alpha=0.7)
ax.set_title("PCA Reduced Data (First Three Components) with Labels")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

# 添加图例
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)
plt.show()