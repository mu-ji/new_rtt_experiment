#n*tau*v <= 9/n i.e. v*n^2*tau <= 9

import numpy as np
import matplotlib.pyplot as plt

# 设置 x 的范围
v = np.linspace(0, 5, 100)
# 定义函数
n = (6000/v)**0.5

# 绘制图形
plt.plot(v, n, label='n = (6000/v)**0.5')
plt.xlabel('speed (m/s)')
plt.ylabel('n')
#plt.axhline(0, color='black', lw=0.5, ls='--')
#plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.savefig('Figures/n-v_relationship.png')
plt.show()