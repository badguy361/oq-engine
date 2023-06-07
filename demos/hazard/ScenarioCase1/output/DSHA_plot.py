import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d,griddata

id = 74
df_site = pd.read_csv(f"Yu2023({id})/sitemesh_{id}.csv",skiprows=[0])
df_gmf = pd.read_csv(f"Yu2023({id})/gmf-data_{id}.csv",skiprows=[0])
x = df_site["lon"]
y = df_site["lat"]
site_id = df_site["site_id"]

gmv_PGA = df_gmf[df_gmf["event_id"]==0]["gmv_PGA"]

plt.scatter(x, y, c=gmv_PGA)
plt.plot((0.0,0.0),(-0.3,0.3),c="r")
plt.colorbar()
# plt.show()

n = 10
xi = np.linspace(x.min(), x.max(), n)
yi = np.linspace(y.min(), y.max(), n)
Xi, Yi = np.meshgrid(xi, yi)

zi = griddata((x, y), gmv_PGA, (Xi, Yi), method='nearest')

plt.contourf(Xi, Yi, zi, levels=20, cmap='viridis')
plt.show()

