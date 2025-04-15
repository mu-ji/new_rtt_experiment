#n*tau*v <= 9/n i.e. v*n^2*tau <= 9

import numpy as np
import matplotlib.pyplot as plt

# 设置 x 的范围
v = np.linspace(0, 5, 100)
# 定义函数
n = (9/(v*1.5e-3))**0.5

resolution = 9/n

# 绘制图形
fig = plt.figure()

ax2=fig.add_subplot(2,1,2)
ax2.plot(v, n)
ax2.fill_between(v, n, color='blue', alpha=0.3, label='n < (9/Tv)**0.5')
ax2.set_xlabel('Speed of reflector (m/s)')
ax2.set_ylabel('Number of packet')
#plt.axhline(0, color='black', lw=0.5, ls='--')
#plt.axvline(0, color='black', lw=0.5, ls='--')

ax2.grid()
ax2.legend()

E = 6.08448e-6
T = 1.5e-3
ax1=fig.add_subplot(2,1,1)
ax1.plot(n, E*n, label = 'Energy consumption')
ax1.set_xlabel('Number of packet')
ax1.set_ylabel('Energy consumption (J)')

ax3 = ax1.twinx()
ax3.plot(n, T*n)
ax3.set_ylabel('Measurement time', color='r')
ax3.tick_params('y', colors='r')

ax1.grid()
ax1.legend()

plt.tight_layout()
plt.savefig('Figures/n-v_relationship.svg')
plt.show()

