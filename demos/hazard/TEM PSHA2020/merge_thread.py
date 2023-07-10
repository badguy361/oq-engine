import pandas as pd
import os
import re

gsim = 'PhungEtAl2020Asc'
area_source = 'S04'
pattern = fr'{gsim}_{area_source}_\d+.csv'
files = os.listdir()
matching_files = [file for file in files if re.match(pattern, file)]

total_csv = [pd.read_csv(thread) for thread in matching_files]
stacked_df = pd.concat([each_csv for each_csv in total_csv], ignore_index=True)

stacked_df.to_csv(f'{gsim}_{area_source}.csv', index=False)