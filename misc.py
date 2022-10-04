#misc.py

import matplotlib.pyplot as plt
import numpy as np

B = np.linspace(8000, 10000, 1000)

#plt.plot(B, np.sin(0.1*B))

B0=9000

g=100

I=20

plt.plot(B, 1-(np.sin(0.03*B)*(I*g**2/((B-B0)**2+g**2)))**2)   

plt.show()  