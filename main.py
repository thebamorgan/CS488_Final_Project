import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# To stop output from being truncated
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

# Read in data
df = pd.read_csv("bike_rental_data.csv")

# ******************************
# PREPROCESSING
# ******************************

# Fix labels
df['season'] = df['season'].map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
df['weather'] = df['weather'].map({1: 'clear', 2: 'mist', 3: 'light rain', 4: 'heavy rain'})

# Convert date time into a structure where it is easier to extract data
df['datetime'] = pd.to_datetime(df['datetime'])
# df['hour'] = df['datetime'].dt.hour
print(df.head())

# Remove boolean values for pairplot because they aren't very helpful
df_pair = df.copy()
df_pair.drop('holiday', axis=1, inplace=True)
df_pair.drop('workingday', axis=1, inplace=True)
# print(df_pair.head())

# Increase the number of colors in the palette
num_colors = len(df_pair['season'].unique())  # Number of unique seasons
palette = sns.color_palette('bright', n_colors=num_colors)

# Plot the pairplot with updated palette
ax = sns.pairplot(df_pair, hue="season", height=2, aspect=0.7, palette=palette)
plt.savefig('pair_plot_no_debug.png')
# plt.show()
# Following a website for this part:
# https://www.analyticsvidhya.com/blog/2021/06/data-cleaning-using-pandas/

# Print total of invalid data per column
print("\nInvalid Data:")
print(df.isna().sum()) # When I ran it everything was zero, good

# Print if there is any duplicated data
print("\nDuplicated Data:")
print(df.duplicated())
print("Sum:", df.duplicated().sum())  # Also zero

# Let's look at the distribution of the data
print(df.describe())

# Since each column is distributed differently, let's compare some scaling methods

# Normalization
df_norm = df.copy()
df_norm.drop('datetime', axis=1, inplace=True)
df_norm.drop('season', axis=1, inplace=True)
df_norm.drop('weather', axis=1, inplace=True)
norm = MinMaxScaler()
df_norm = pd.DataFrame(norm.fit_transform(df_norm))
df_norm.columns = ['holiday', 'workingday', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
print(df_norm['workingday'])

fig, axes = plt.subplots(1, 2, figsize=(20, 16))
sns.boxplot(data=(df_norm['workingday'], df_norm['count']), ax=axes[0])
sns.boxplot(data=(df_norm['holiday'], df_norm['count']), ax=axes[1])
#sns.boxplot(data=df_norm, x='season', y='count', ax=axes[1, 0])
#sns.boxplot(data=df_norm, x='weather', y='count', ax=axes[1, 1])
plt.savefig('normalization.png')



