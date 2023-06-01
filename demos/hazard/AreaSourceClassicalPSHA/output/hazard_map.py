import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d,griddata
import sys
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)

# df = pd.read_csv("AbrahamsonEtAl2014(27)/hazard_map-mean_27.csv",skiprows=[0])
df = pd.read_csv("Yu2023(19)/hazard_map-mean_19.csv",skiprows=[0])
# df = pd.read_csv("result.csv",skiprows=[0])
x = df["lon"]
y = df["lat"]
z = df["PGA-0.1"]

plt.scatter(x,
            y,
            c=z)
plt.colorbar()
plt.show()

n = 10
xi = np.linspace(x.min(), x.max(), n)
yi = np.linspace(y.min(), y.max(), n)
Xi, Yi = np.meshgrid(xi, yi)

# zi = interp2d(x, y, z, kind='cubic')(xi, yi)

zi = griddata((x, y), z, (Xi, Yi), method='nearest')

plt.contourf(Xi, Yi, zi, levels=20, cmap='viridis')
plt.colorbar()
# plt.xlim(-0.75, 0.75)
plt.show()