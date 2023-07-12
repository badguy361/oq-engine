import pandas as pd
import os
import re

gsim1 = 'Lin2011hanging'
gsim2 = 'ChaoEtAl2020Asc'
gsim3 = 'Lin2011foot'
area_source = 'S04'
pattern = fr'{gsim1}|{gsim2}|{gsim3}_{area_source}_\d+.csv'
files = os.listdir()
matching_files = [file for file in files if re.match(pattern, file)]

total_csv = [pd.read_csv(thread) for thread in matching_files] # read all csv
stacked_df = pd.concat([each_csv for each_csv in total_csv], ignore_index=True)

stacked_df.to_csv(f'TotalGsim_{area_source}.csv', index=False)