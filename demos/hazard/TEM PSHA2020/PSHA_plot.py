import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d,griddata

id = 2
folder = 'Yu2023'
hazard_map = pd.read_csv(f"{folder}({id})/hazard_map-mean_{id}.csv",skiprows=[0])
hazard_curve = pd.read_csv(f"{folder}({id})/hazard_curve-mean-PGA_{id}.csv",skiprows=[0])
hazard_uhs = pd.read_csv(f"{folder}({id})/hazard_uhs-mean_{id}.csv",skiprows=[0])

y = hazard_curve.iloc[231]
pga = [float(_[4:]) for _ in hazard_curve.columns[3:]]
fig = plt.figure()
plt.scatter(pga,y[3:])
plt.yscale("log")
plt.xscale("log")
plt.show()

