import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

df = pd.read_csv("hazard_map-mean_6.csv",skiprows=[0])
x = df["lon"]
y = df["lat"]
z = df["PGA-0.1"]

plt.scatter(x,
            y,
            c=z)
plt.colorbar()

n = 100
xi = np.linspace(x.min(), x.max(), n)
yi = np.linspace(y.min(), y.max(), n)
Xi, Yi = np.meshgrid(xi, yi)
zi = interp2d(x, y, z, kind='cubic')(xi, yi)
plt.contourf(Xi, Yi, zi, levels=20, cmap='viridis')