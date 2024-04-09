import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def plot_learning_curve(est_arr, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), title=" "):
    """
    **Found on sklearn website and modified**
    https://scikit-learn.org/0.15/auto_examples/plot_learning_curve.html

    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    test = []
    train = []
    colors = []

    # Loop through each estimator and apply the learning curve function on them
    for c in est_arr:
        estimator = est_arr[c][0]
        color = est_arr[c][1]
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                                train_sizes=train_sizes, shuffle=True, verbose=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test.append(test_scores_mean * 100)
        train.append(train_scores_mean * 100)
        colors.append(color)

    # Plot the learning curve scores for each estimator as a line graph
    fig, (s_test, s_train) = plt.subplots(1, 2)
    fig.set(figheight=6, figwidth=10)
    s_test.set_box_aspect(1)
    s_test.set_title("Model Evaluation - Cross Validation Accuracy")
    s_test.set_ylabel("Overall Classification Accuracy")
    s_test.set_xlabel("% of Training Examples")
    s_train.set_box_aspect(1)
    s_train.set_title("Model Evaluation - Training Accuracy")
    s_train.set_ylabel("Training Recall Accuracy")
    s_train.set_xlabel("% of Training Examples")

    class_num = []
    for i in range(len(colors)):
        class_num.append(i)

    for color, i, label in zip(colors, class_num, est_arr):
        s_test.plot(
            [10, 20, 30, 40, 50], test[i], color=color, label=label, marker='o'
        )

        s_train.plot(
            [10, 20, 30, 40, 50], train[i], color=color, label=label, marker='o'
        )

    s_train.legend(loc="best")
    s_test.legend(loc="best")
    fig.suptitle(title, y=0.92)
    return plt, train_scores

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
target = np.swapaxes(target, 0, 1)
target = pd.DataFrame(target, columns=['season', 'holiday', 'workingday', 'weather'])
df.drop(['season', 'holiday', 'workingday', 'weather'], axis=1, inplace=True)

# Assigning labels for Season and Weather
#target['season'] = target['season'].map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
#target['weather'] = target['weather'].map({1: 'clear', 2: 'mist', 4: 'light rain', 4: 'heavy rain'})

print(df.head())

df_pair = df.copy()

# Increase the number of colors in the palette
num_colors = len(target['season'].unique())  # Number of unique seasons
palette = sns.color_palette('bright', n_colors=num_colors)

# Plot the pairplot with updated palette
#ax = sns.pairplot(df_pair, height=2, aspect=0.7, palette=palette)
#plt.savefig('pair_plot_no_debug.png')

# ******************************
# CORRELATION MATRIX
# ******************************

# Visualize features as a heatmap
cor_eff=df.corr()
plt.figure(figsize=(6,6))
sns.heatmap(cor_eff,linecolor="white",linewidths=1,annot=True)
plt.savefig("full_corr_matrix")

# Plot the lower half of the correlation matrix
fig, ax = plt.subplots(figsize=(6,6))
# Compute the correlation matrix
mask=np.zeros_like(cor_eff)
plt.savefig("lower_corr_matrix")

# mask = 0: display the correlation matrix
# mask = 1: display the unique lower triangular values
#mask[np.triu_indices_from(mask)] = 0
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(cor_eff,linecolor="white",linewidths=1,mask=mask,ax=ax,annot=True)

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
#df_final = pd.concat(objs=(df_final, target), axis=1)

# This is kinda useless now
'''
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
sns.boxplot(data=df_final, x='workingday', y='count', ax=axes[0, 0])
sns.boxplot(data=df_final, x='holiday', y='count', ax=axes[0, 1])
sns.boxplot(data=df_final, x='season', y='count', ax=axes[1, 0])
sns.boxplot(data=df_final, x='weather', y='count', ax=axes[1, 1])
plt.show()
'''
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

'''
# Assigning X and Y for PCA
x_pca = df_final
y_pca = target['season']

# PCA
pca = PCA()
pca = pca.fit(x_pca)
ev = pca.explained_variance_ratio_

# Explained Variance Plot - PCA
plt.figure(figsize=(8, 6))
plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], list(ev * 100), label='Principal Components', color='c')
plt.legend()
plt.xlabel('Principal Components')
pc = []
for i in range(12):
    pc.append('PC-' + str(i + 1))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], pc, fontsize=8, rotation=30)
plt.ylabel('Variance Ratio')
plt.title('PCA Variance Ratio of Bike Rental Dataset')
plt.savefig('PCA_class_variance')
# plt.show()

# PCA Dimensionality Reduction
pca = PCA()
pc = pca.fit_transform(x_pca)

x1 = x_pca.transpose()
X_pca = np.matmul(pc, x1)


# Reformat reduced data
X_pca_df = pd.DataFrame(data=pc)
target_df = pd.DataFrame(data=target['season'])
X_pca_df.columns = ['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9', 'PC-10', 'PC-11', 'PC-12']

# 2D PCA Reduction Plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('PC-1', fontsize=15)
ax.set_ylabel('PC-2', fontsize=15)
ax.set_title('PCA on Bike Rental Dataset', fontsize=20)
class_num = [1, 2, 3, 4]
colors = ['turquoise', 'gold', 'darkorange', 'mediumpurple']
for target, color in zip(class_num, colors):
    indicesToKeep = y_pca == target #This doesn't make sense anymore?
    ax.scatter(X_pca_df.loc[indicesToKeep, 'PC-1'], #UGGGHHH
               X_pca_df.loc[indicesToKeep, 'PC-2'],
               c=color, s=9)
ax.legend(class_num)
ax.grid()
plt.show()
'''

# Count over a day for working days vs not
#sns.lineplot(data=df_final, x='hour', y='count', hue='workingday')
#plt.show()

# Assigning X and Y for LDA
x_lda = df_final
y_lda = target['season']

# LDA
lda = LinearDiscriminantAnalysis()
lda = lda.fit(X=x_lda, y=y_lda)
ev = lda.explained_variance_ratio_

# Explained Variance Plot - LDA
plt.figure(figsize=(8, 6))
plt.bar([1, 2, 3], list(ev * 100), label='Linear Discriminants', color='c')
plt.legend()
plt.xlabel('Linear Discriminants')
ld = []
for i in range(3):
    ld.append('LD-' + str(i + 1))
plt.xticks([1, 2, 3], ld, fontsize=8, rotation=30)
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
colors = ['turquoise', 'gold', 'darkorange', 'mediumpurple']
#markerm = ['o', 'o', 'o', 'o', 'o', 'o', 'o', '+', '+', '+', '+', '+', '+', '+', '*', '*']
for t, color in zip(class_num, colors):
    indicesToKeep = y_lda == t
    ax.scatter(x_lda[indicesToKeep, 0],
               x_lda[indicesToKeep, 1],
               c=color, s=9)
ax.legend(['Spring', 'Summer', 'Fall', 'Winter'])
ax.grid()
plt.savefig('LDA_scatterplot_season')
#plt.show()

# Supervised Classification w/o reduction
classifier_labels = {"SVM - RBF": (SVC(kernel="rbf", random_state=1), "green"),
                     "SVM - Poly": (SVC(kernel="poly", random_state=1), "darkorange"),
                     "SVM - Linear": (SVC(kernel="linear"), "blue"),
                     "Gaussian NB": (GaussianNB(), "purple"),
                     "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=1), "red"),
                     "Random Forest": (RandomForestClassifier(random_state=1), "gold"),
                     "kNN": (KNeighborsClassifier(n_neighbors=5), "gray")}


# **WARNING**, this section of the code can take up to 10 min to run
fig1, normal_scores = plot_learning_curve(est_arr=classifier_labels, X=df_final, y=target['season'], train_sizes=np.linspace(start=0.1, stop=0.5, num=5), cv=5, n_jobs=1,
                   title="Supervised Classification of Bike Rental Dataset Without Dimensionality Reduction")
# plt.show()
plt.savefig('classification_accuracy')


# LDA
fig3, lda_scores = plot_learning_curve(est_arr=classifier_labels, X=x_lda, y=target['season'], train_sizes=np.linspace(start=0.1, stop=0.5, num=5), cv=5, n_jobs=1,
                   title="Supervised Classification of Bike Rental Dataset With LDA Dimensionality Reduction")
#plt.show()
plt.savefig('classification_accuracy_LDA')

# Using classification data, test best classifier (Random Forest)
x_train, x_test, y_train, y_test = train_test_split(x_lda, y_lda, train_size=0.2, random_state=1)
rf = RandomForestClassifier(random_state=1)
rf.fit(x_train, y_train)
# Get predicted values to calculate RMSE and MSE
y_pred = rf.predict(x_test)

print("\nTraining Results")
print("-----------------------------------------------------")
print("Coefficient of Determination:", round(r2_score(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))

# Predict new data (I want to be able to predict count instead of season, hmm...)
pred = rf.predict(x_test)
labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
print("\nTraining Predicted Values")
print("-----------------------------------------------------")
print("Predicted Season:", labels.get(pred[0]))
print("Actual Season:", labels.get(y_test.values[0]))



