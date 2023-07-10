import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
import pygmt

fault_data = [
    120.3026, 22.8980, 120.3023, 22.9025, 120.3021, 22.9070, 120.3018, 22.9115,
    120.3015, 22.9160, 120.3014, 22.9205, 120.3013, 22.9250, 120.3012, 22.9296,
    120.3010, 22.9341, 120.3009, 22.9386, 120.3008, 22.9431, 120.3006, 22.9476,
    120.3005, 22.9521, 120.3003, 22.9566, 120.3001, 22.9611, 120.3002, 22.9657,
    120.3010, 22.9701, 120.3018, 22.9746, 120.3027, 22.9790, 120.3036, 22.9834,
    120.3048, 22.9878, 120.3061, 22.9921, 120.3073, 22.9965, 120.3090, 23.0008,
    120.3107, 23.0050, 120.3127, 23.0091, 120.3138, 23.0135, 120.3147, 23.0179,
    120.3162, 23.0222, 120.3185, 23.0262, 120.3218, 23.0295, 120.3250, 23.0329,
    120.3276, 23.0367, 120.3295, 23.0408, 120.3303, 23.0453, 120.3309, 23.0498,
    120.3317, 23.0542, 120.3321, 23.0587, 120.3320, 23.0632, 120.3318, 23.0677,
    120.3317, 23.0722, 120.3320, 23.0767, 120.3331, 23.0811, 120.3339, 23.0854,
    120.3384, 23.0872, 120.3429, 23.0889, 120.3474, 23.0906, 120.3514, 23.0931,
    120.3543, 23.0967, 120.3558, 23.1009, 120.3567, 23.1053, 120.3574, 23.1098,
    120.3579, 23.1143, 120.3580, 23.1188, 120.3578, 23.1233, 120.3570, 23.1278,
    120.3558, 23.1321, 120.3545, 23.1364, 120.3530, 23.1407, 120.3508, 23.1447,
    120.3498, 23.1490, 120.3482, 23.1533, 120.3457, 23.1571, 120.3423, 23.1603,
    120.3383, 23.1629, 120.3342, 23.1653, 120.3300, 23.1676, 120.3259, 23.1700,
    120.3225, 23.1733, 120.3199, 23.1771, 120.3199, 23.1815, 120.3199, 23.1857
]

#'AbrahamsonEtAl2014','Lin2009','Chang2023'
id = 97
folder = 'AbrahamsonEtAl2014'
df_site = pd.read_csv(f"100result/{folder}({id})/sitemesh_{id}.csv", skiprows=[0])
df_gmf = pd.read_csv(f"100result/{folder}({id})/gmf-data_{id}.csv", skiprows=[0])
df_total = df_gmf.merge(df_site, how='left', on='site_id')
df_total = df_total.groupby("site_id").mean()

gmv_PGA = df_total["gmv_PGA"]
x = df_total["lon"]
y = df_total["lat"]

fig = pygmt.Figure()
region = [119.5, 122.5, 21.5, 25.5]

fig.basemap(region=region,
            projection="M12c",
            frame=["af", f"WSne+tScenario Hazard Analysis gmm type: {folder}"])
fig.coast(land="gray", water="gray", shorelines="1p,black")
pygmt.makecpt(cmap="turbo", series=(0, 1.5))
fig.plot(x=fault_data[::2], y=fault_data[1::2], pen="thick,red")
fig.plot(x=x, y=y, style="c0.2c", cmap=True, color=gmv_PGA)
fig.colorbar(frame=["x+lPGA(g)"])
fig.savefig(f"100result/{folder}({id})/gmv_PGA.png", dpi=300)
fig.show()
