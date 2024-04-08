import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

# These are the possible targets because they are labeled
target = np.array((df['season'], df['holiday'], df['workingday'], df['weather']))
target = target.reshape(-1, 4)
target = pd.DataFrame(target, columns=['season', 'holiday', 'workingday', 'weather'])
df.drop(['season', 'holiday', 'workingday', 'weather'], axis=1, inplace=True)

# Assigning labels for Season and Weather
#target['season'] = target['season'].map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
#target['weather'] = target['weather'].map({1: 'clear', 2: 'mist', 4: 'light rain', 4: 'heavy rain'})

# Count is both of these combined, so maybe we should drop them?
# df.drop('registered', axis=1, inplace=True)
# df.drop('casual', axis=1, inplace=True)

print(df.head())

# Remove boolean values for pairplot because they aren't very helpful
df_pair = df.copy()
# df_pair.drop('holiday', axis=1, inplace=True)
# df_pair.drop('workingday', axis=1, inplace=True)

# Increase the number of colors in the palette
num_colors = len(target['season'].unique())  # Number of unique seasons
palette = sns.color_palette('bright', n_colors=num_colors)

# Plot the pairplot with updated palette
#ax = sns.pairplot(df_pair, height=2, aspect=0.7, palette=palette)
#plt.savefig('pair_plot_no_debug.png')


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
df_final.columns = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count', 'hour']

# This is kinda useless now
# fig, axes = plt.subplots(1, 2, figsize=(20, 16))
# sns.boxplot(data=(df_final['workingday'], target['count']), ax=axes[0])
# sns.boxplot(data=(df_final['holiday'], target['count']), ax=axes[1])
# plt.savefig('normalization.png')

# ******************************
# FEATURE SELECTION
# ******************************

# Remove outliers (we can scrap this if we need to)
q1 = df_final['count'].quantile(0.25)
q3 = df_final['count'].quantile(0.75)
iqr = q3-q1
outliers = df_final.index[((df_final['count'] < q1-(1.5 * iqr)) | (df_final['count'] > q3+(1.5 * iqr)))].tolist()
target = target.drop(index=outliers, axis=0)
df_final = df_final.drop(index=outliers, axis=0)

# Assigning X and Y for PCA
x_pca = df_final
y_pca = target['season']

# PCA
pca = PCA()
pca = pca.fit(x_pca)
ev = pca.explained_variance_ratio_

# Explained Variance Plot - PCA
plt.figure(figsize=(8, 6))
plt.bar([1, 2, 3, 4, 5, 6, 7, 8], list(ev * 100), label='Principal Components', color='c')
plt.legend()
plt.xlabel('Principal Components')
pc = []
for i in range(8):
    pc.append('PC-' + str(i + 1))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], pc, fontsize=8, rotation=30)
plt.ylabel('Variance Ratio')
plt.title('PCA Variance Ratio of Bike Rental Dataset')
plt.savefig('PCA_class_variance')
# plt.show()

# PCA Dimensionality Reduction
pca = PCA()
pc = pca.fit_transform(x_pca)
x1 = x_pca.transpose()
X_pca = np.matmul(x1, pc)

# Reformat reduced data
X_pca_df = pd.DataFrame(data=X_pca)
X_pca_df.columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8']

# 2D PCA Reduction Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC-1', fontsize=15)
ax.set_ylabel('PC-2', fontsize=15)
ax.set_title('PCA on Bike Rental Dataset', fontsize=20)
class_num = [1, 2, 3, 4]
colors = ['r', 'k', 'b', 'g']
for target, color in zip(class_num, colors):
    indicesToKeep = y_pca == target
    ax.scatter(X_pca_df.loc[indicesToKeep, 'PC-1'], #UGGGHHH
               X_pca_df.loc[indicesToKeep, 'PC-2'],
               c=color, s=9)
ax.legend(class_num)
ax.grid()
plt.show()

# Assigning X and Y for LDA
x_lda = df_final
y_lda = target['season'] # I don't think count should be the Y, It should probably be weather or season

# LDA
lda = LinearDiscriminantAnalysis()
lda = lda.fit(X=x_lda, y=y_lda)
ev = lda.explained_variance_ratio_

# Explained Variance Plot - LDA
plt.figure(figsize=(8, 6))
plt.bar([1, 2, 3, 4], list(ev * 100), label='Linear Discriminants', color='c')
plt.legend()
plt.xlabel('Linear Discriminants')
ld = []
for i in range(4):
    ld.append('LD-' + str(i + 1))
plt.xticks([1, 2, 3, 4], ld, fontsize=8, rotation=30)
plt.ylabel('Variance Ratio')
plt.title('LDA Variance Ratio of Bike Rental Dataset')
plt.savefig('LDA_class_variance')
# plt.show()

# LDA Dimensionality Reduction
lda = LinearDiscriminantAnalysis(n_components=2)
x_lda = lda.fit_transform(X=x_lda, y=y_lda)

# Plot LDA scatterplot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('LDA on Bike Rental Dataset - Season', fontsize=20)
class_num = [1, 2, 3, 4]
colors = ['r', 'k', 'b', 'g']
#markerm = ['o', 'o', 'o', 'o', 'o', 'o', 'o', '+', '+', '+', '+', '+', '+', '+', '*', '*']
for target, color in zip(class_num, colors):
    indicesToKeep = y_lda == target
    ax.scatter(x_lda[indicesToKeep, 0],
               x_lda[indicesToKeep, 1],
               c=color, s=9)
ax.legend(class_num)
ax.grid()
plt.savefig('LDA_scatterplot')
# plt.show()




