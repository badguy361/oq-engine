import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
import pygmt

fault_data = [
    120.9319, 24.6990, 120.9335, 24.7033, 120.9350, 24.7076, 120.9365, 24.7119,
    120.9389, 24.7157, 120.9427, 24.7185, 120.9469, 24.7209, 120.9512, 24.7232,
    120.9552, 24.7257, 120.9593, 24.7282, 120.9635, 24.7307, 120.9676, 24.7332,
    120.9716, 24.7358, 120.9757, 24.7383, 120.9797, 24.7410, 120.9832, 24.7442,
    120.9866, 24.7474, 120.9904, 24.7504, 120.9944, 24.7529, 120.9986, 24.7554,
    121.0027, 24.7579, 121.0067, 24.7605, 121.0107, 24.7632, 121.0147, 24.7658,
    121.0188, 24.7683, 121.0233, 24.7702, 121.0278, 24.7720, 121.0322, 24.7742,
    121.0363, 24.7766, 121.0403, 24.7793, 121.0444, 24.7818, 121.0485, 24.7844,
    121.0529, 24.7865, 121.0577, 24.7875, 121.0625, 24.7886, 121.0673, 24.7897,
    121.0721, 24.7903, 121.0770, 24.7912, 121.0819, 24.7916, 121.0868, 24.7917,
    121.0917, 24.7916, 121.0967, 24.7915, 121.1016, 24.7915, 121.1066, 24.7915,
    121.1115, 24.7920, 121.1164, 24.7918, 121.1213, 24.7919, 121.1262, 24.7919,
    121.1311, 24.7922, 121.1361, 24.7919, 121.1409, 24.7911, 121.1455, 24.7895,
    121.1502, 24.7880, 121.1547, 24.7861, 121.1592, 24.7843, 121.1637, 24.7824,
    121.1681, 24.7804, 121.1723, 24.7779
]

#'AbrahamsonEtAl2014','Lin2009','Chang2023'
id = 95
folder = 'Chang2023'
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
