import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
pd.set_option('display.max_columns', 100)

df_co2_emissions = pd.read_csv('./datasets/raw/co2_emissions/co2-emissions-per-capita.csv')
df_co2_emissions.head()