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

# Convert date time into a structure where it is easier to extract data
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df.drop('datetime', axis=1, inplace=True)
print(df.head())

# Remove boolean values for pairplot because they aren't very helpful
df_pair = df.copy()
df_pair.drop('holiday', axis=1, inplace=True)
df_pair.drop('workingday', axis=1, inplace=True)

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

# Normalization
norm = MinMaxScaler()
df_final = pd.DataFrame(norm.fit_transform(df))
df_final.columns = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual',
                    'registered', 'count', 'hour']

# Assigning labels for Season and Weather
df_final['season'] = df_final['season'].map({0: 'spring', (1/3): 'summer', (2/3): 'fall', 1: 'winter'})
df_final['weather'] = df_final['weather'].map({0: 'clear', (1/3): 'mist', (2/3): 'light rain', 1: 'heavy rain'})

fig, axes = plt.subplots(1, 2, figsize=(20, 16))
sns.boxplot(data=(df_final['workingday'], df_final['count']), ax=axes[0])
sns.boxplot(data=(df_final['holiday'], df_final['count']), ax=axes[1])
#sns.boxplot(data=df_norm, x='season', y='count', ax=axes[1, 0])
#sns.boxplot(data=df_norm, x='weather', y='count', ax=axes[1, 1])
plt.savefig('normalization.png')



