import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("bike_rental_data.csv")
df['season'] = df['season'].map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
#df['weather'] = df['weather'].map({1: 'clear', 2: 'mist', 3: 'light rain', 4: 'heavy rain'})
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
print(df.head())

df_pair = df.copy()
df_pair.drop('holiday', axis=1, inplace=True)
df_pair.drop('workingday', axis=1, inplace=True)
print(df_pair.head())

ax = sns.pairplot(df_pair, hue="season", height=1.75, aspect=0.7, palette=sns.color_palette('bright'))
print("hello")

