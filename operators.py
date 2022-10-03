#models.py

import numpy as np

# Single spin operators
opIz = np.array([[1/2, 0], [0, -1/2]])
opIp = np.array([[0, 1], [0, 0]])
opIm = np.array([[0, 0], [1, 0]])
mx2 = np.identity(2)
mx3=np.kron(opIz,mx2)

print("Matrix1 =\n",opIz)
print("\nMatrix2 =\n",mx2)
print("\nKronecker Product =\n",mx3)