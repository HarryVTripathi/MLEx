import numpy as np
from matplotlib import pyplot as plt


x, y = np.arange(-10, 11, 2), np.arange(-10, 11, 2)
U, V = np.meshgrid(x, y)

X = V**3 -9*V
Y = U**3 -9*V

fig1, ax1 = plt.subplots()
Q = ax1.quiver(U, V, X, Y, units='width')
qk = ax1.quiverkey(Q, 1, 1, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')
plt.show()