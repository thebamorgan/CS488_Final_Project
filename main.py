import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import learning_curve, train_test_split, KFold, StratifiedShuffleSplit, ShuffleSplit
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
    est_arr : dict with structure {"classifier label": (classifier function, color)}
        Estimators to be used on the data. Color corresponds to the estimator's color on the graph

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

def plot_regression_curves(reg_arr, X, y, title=" "):
    """
    **Found on sklearn website and modified**
    https://scikit-learn.org/0.15/auto_examples/plot_learning_curve.html

    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    reg_arr : dict with structure {"regression label": (regression function, color)}
        Estimators to be used on the data. Color corresponds to the estimator's color on the graph

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    """
    test = []
    colors = []

    # Loop through each estimator and apply the learning curve function on them
    for c in reg_arr:
        print("Starting calculations for", reg_arr[c])
        estimator = reg_arr[c][0]
        color = reg_arr[c][1]
        test_scores = []

        for i in range(1, 9):
            score = []
            ss = ShuffleSplit(n_splits=5, train_size=(i/10), random_state=1)
            for i, (train_index, test_index) in enumerate(ss.split(X=X, y=y)):
                x_train = X[train_index]
                y_train = y[train_index]
                x_test = X[test_index]
                y_test = y[test_index]

                estimator.fit(x_train, y_train)
                # Get predicted values to calculate RMSE and MSE
                y_pred = estimator.predict(x_test)

                correct_count = 0
                y_pred = np.round(y_pred).astype(int)
                for i in range(len(y_pred)):
                    if (y_test[i] >= (y_pred[i] - 25) and y_test[i] <= (y_pred[i]  + 25)):
                        correct_count += 1

                # Calculate accuracy of predictions
                accuracy = (correct_count / len(y_pred)) * 100
                score.append(accuracy)
            test_scores.append(np.mean(score))

        test.append(test_scores)
        colors.append(color)

    # Plot the learning curve scores for each estimator as a line graph
    fig, (s_test) = plt.subplots(1, 1)
    fig.set(figheight=6, figwidth=10)
    s_test.set_box_aspect(1)
    s_test.set_ylabel("Overall Classification Accuracy")
    s_test.set_xlabel("% of Training Examples")

    class_num = []
    for i in range(len(colors)):
        class_num.append(i)

    for color, i, label in zip(colors, class_num, reg_arr):

        s_test.plot(
            [10, 20, 30, 40, 50, 60, 70, 80], test[i], color=color, label=label, marker='o'
        )

    s_test.legend(loc="best")
    fig.suptitle(title, y=0.92)
    return plt, test

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
df['month'] = df['datetime'].dt.month
df.drop('datetime', axis=1, inplace=True)

# Also try using casual or registered columns instead of count

# These are the possible targets because they are labeled
target = np.array((df['season'], df['holiday'], df['workingday'], df['weather']))
target = np.swapaxes(target, 0, 1)
target = pd.DataFrame(target, columns=['season', 'holiday', 'workingday', 'weather'])
df.drop(['season', 'holiday', 'workingday', 'weather'], axis=1, inplace=True)

print(df.head())

df_pair = df.copy()

# Increase the number of colors in the palette
num_colors = len(target['season'].unique())  # Number of unique seasons
palette = sns.color_palette('bright', n_colors=num_colors)
'''
# Plot the pairplot with updated palette
ax = sns.pairplot(df_pair, height=2, aspect=0.7)
plt.savefig('pair_plot_no_debug.png')

# ******************************
# CORRELATION MATRIX
# ******************************

# Visualize features as a heatmap
cor_eff = df.corr()
plt.figure(figsize=(9, 9))
sns.heatmap(cor_eff, linecolor="white", linewidths=1, annot=True)
plt.savefig("full_corr_matrix")
# plt.show()

# Plot the lower half of the correlation matrix
fig, ax = plt.subplots(figsize=(9, 9))

# Compute the correlation matrix
mask = np.zeros_like(cor_eff)

# mask = 0: display the correlation matrix
# mask = 1: display the unique lower triangular values
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(cor_eff, linecolor="white", linewidths=1, mask=mask, ax=ax, annot=True)
plt.savefig("lower_corr_matrix")
# plt.show()
'''
# Print total of invalid data per column
print("\nInvalid Data:")
print(df.isna().sum()) # When I ran it everything was zero, good

# Print if there is any duplicated data
print("\nDuplicated Data:")
print(df.duplicated())
print("Sum:", df.duplicated().sum())  # Also zero

# Let's look at the distribution of the data
print(df.describe())

# Histogram for distribution of count column
plt.clf()
df['count'].plot.hist(bins=10, legend=None)
plt.title("Distribution of Bike Rental Counts per Hour")
plt.xlabel("Bikes Rented per Hour")
plt.savefig("count_histogram")
#plt.show()

# Divide count column so it can be used as a label
df_count = pd.DataFrame(data=df['count'])
target['count'] = pd.cut(x=df['count'], bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(target['count'].value_counts())

# Normalization
norm = MinMaxScaler()
df_final = pd.DataFrame(norm.fit_transform(df))
df_final.columns = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count', 'hour', 'month']

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
df_count = df_count.drop(index=outliers, axis=0)
df_final = pd.concat(objs=[df_final, target['season'], target['holiday'], target['workingday'], target['weather']], axis=1)

# Drop target values
df_final.drop(['count'], axis=1, inplace=True)

discrete_features = df_final.dtypes == int
scores = mutual_info_regression(df_final, target['count'], discrete_features=discrete_features, random_state=42)
scores = pd.Series(scores, name='Mutual Information', index=df_final.columns)
# scores = scores.sort_values(ascending=False)
scores = scores.sort_values(ascending=True)
width = np.arange(len(scores))
ticks = list(scores.index)
plt.figure(figsize=(8, 6))
plt.barh(width, scores)
plt.yticks(width, ticks)
plt.title("Yulu Dataset Mutual Information Scores")
plt.savefig("mutual_info_scores")
#plt.show()

# Assigning X and Y for LDA
x_lda = df_final
y_lda = target['count']

df_noreg = df_final.drop(['registered', 'casual'], axis=1)
df_noreg.drop(['holiday', 'workingday', 'weather', 'windspeed'], axis=1, inplace=True)
x_lda_noreg = df_noreg

# LDA
lda = LinearDiscriminantAnalysis()
lda = lda.fit(X=x_lda_noreg, y=y_lda)
ev = lda.explained_variance_ratio_

# Explained Variance Plot - LDA
plt.figure(figsize=(8, 6))
plt.bar([1, 2, 3, 4, 5, 6], list(ev * 100), label='Linear Discriminants', color='c')
plt.legend()
plt.xlabel('Linear Discriminants')
ld = []
for i in range(6):
    ld.append('LD-' + str(i + 1))
plt.xticks([1, 2, 3, 4, 5, 6], ld, fontsize=8, rotation=30)
plt.ylabel('Variance Ratio')
plt.title('LDA Variance Ratio of Yulu Dataset')
plt.savefig('LDA_class_variance_count')
#plt.show()

# LDA Dimensionality Reduction
lda = LinearDiscriminantAnalysis(n_components=2)
x_lda = lda.fit_transform(X=x_lda, y=y_lda)

# Plot LDA scatterplot
plt.clf()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('LDA on Yulu Dataset', fontsize=20)
class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
colors = ['r', 'b', 'g', 'y', 'orange', 'c', 'k', 'pink', 'brown', 'gray']
for t, color in zip(class_num, colors):
    indicesToKeep = y_lda == t
    ax.scatter(x_lda[indicesToKeep, 0],
               x_lda[indicesToKeep, 1],
               c=color, s=9)
ax.legend(['0 to 100', '100 to 200', '200 to 300', '300 to 400', '400 to 500', '500 to 600', '600 to 700', '700 to 800',
           '800 to 900', '900 to 1000'], title="Number of Bikes Rented")
ax.grid()
plt.savefig('LDA_scatterplot_count')
#plt.show()

lda = LinearDiscriminantAnalysis(n_components=2)
x_lda_noreg = lda.fit_transform(X=x_lda_noreg, y=y_lda)

# Plot LDA scatterplot no registered and casual
plt.clf()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('LDA on Yulu Dataset', fontsize=20)
class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
colors = ['r', 'b', 'g', 'y', 'orange', 'c', 'k', 'pink', 'brown', 'gray']
for t, color in zip(class_num, colors):
    indicesToKeep = y_lda == t
    ax.scatter(x_lda_noreg[indicesToKeep, 0],
               x_lda_noreg[indicesToKeep, 1],
               c=color, s=9)
ax.legend(['0 to 100', '100 to 200', '200 to 300', '300 to 400', '400 to 500', '500 to 600', '600 to 700', '700 to 800',
           '800 to 900', '900 to 1000'], title="Number of Bikes Rented")
ax.grid()
plt.savefig('LDA_scatterplot_count_noreg')
#plt.show()

# ******************************
# CLASSIFICATION
# ******************************

# Supervised Classification
classifier_labels = {"SVM - RBF": (SVC(kernel="rbf", random_state=1), "green"),
                     "SVM - Poly": (SVC(kernel="poly", random_state=1), "darkorange"),
                     "SVM - Linear": (SVC(kernel="linear"), "blue"),
                     "Gaussian NB": (GaussianNB(), "purple"),
                     "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=1), "red"),
                     "Random Forest": (RandomForestClassifier(random_state=1), "gold"),
                     "kNN": (KNeighborsClassifier(n_neighbors=5), "gray")}
'''
# **WARNING**, this section of the code can take up to 10 min to run
fig1, normal_scores = plot_learning_curve(est_arr=classifier_labels, X=df_final, y=target['count'], train_sizes=np.linspace(start=0.1, stop=0.5, num=5), cv=5, n_jobs=1,
                   title="Supervised Classification of Yulu Dataset Without Dimensionality Reduction")
plt.savefig('classification_accuracy_count')
plt.show()

# LDA
fig3, lda_scores = plot_learning_curve(est_arr=classifier_labels, X=x_lda, y=target['count'], train_sizes=np.linspace(start=0.1, stop=0.5, num=5), cv=5, n_jobs=1,
                   title="Supervised Classification of Yulu Dataset With LDA Dimensionality Reduction")
plt.savefig('classification_accuracy_LDA_count')
plt.show()

# Without registered and casual
fig1, normal_scores = plot_learning_curve(est_arr=classifier_labels, X=df_noreg, y=target['count'], train_sizes=np.linspace(start=0.1, stop=0.5, num=5), cv=5, n_jobs=1,
                   title="Supervised Classification of Yulu Dataset Without Dimensionality Reduction")
plt.savefig('classification_accuracy_count_no_reg')
plt.show()

# LDA
fig3, lda_scores = plot_learning_curve(est_arr=classifier_labels, X=x_lda_noreg, y=target['count'], train_sizes=np.linspace(start=0.1, stop=0.5, num=5), cv=5, n_jobs=1,
                   title="Supervised Classification of Yulu Dataset With LDA Dimensionality Reduction")
plt.savefig('classification_accuracy_LDA_count_no_reg')
plt.show()
'''
# ******************************
# SCORING
# ******************************

# Using classification data, test best classifier
x_train, x_test, y_train, y_test = train_test_split(x_lda, y_lda, train_size=0.5, random_state=1, shuffle=True)

rbf = SVC(kernel="rbf", random_state=1)
rbf.fit(x_train, y_train)

# Get predicted values to calculate RMSE and MSE
y_pred = rbf.predict(x_test)

print("\nTraining Results with Registered and Casual")
print("-----------------------------------------------------")
print("Coefficient of Determination:", round(r2_score(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))

# Check predictions
correct_count = 0
for i in range(len(y_test.values)):
    if y_test.values[i] == y_pred[i]:
        correct_count += 1

# Calculate accuracy of predictions
accuracy = (correct_count/len(y_test.values)) * 100

print("Prediction Accuracy: ", round(accuracy, 2), "%")

# No registered and casual
# Using classification data, test best classifier
x_train, x_test, y_train, y_test = train_test_split(x_lda_noreg, y_lda, train_size=0.2, random_state=1, shuffle=True)

rbf = SVC(kernel="rbf", random_state=1)
rbf.fit(x_train, y_train)

# Get predicted values to calculate RMSE and MSE
y_pred = rbf.predict(x_test)

print("\n\nTraining Results without Registered and Casual")
print("-----------------------------------------------------")
print("Coefficient of Determination:", round(r2_score(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))


# Check predictions
correct_count = 0
for i in range(len(y_test.values)):
    if y_test.values[i] == y_pred[i]:
        correct_count += 1

# Calculate accuracy of predictions
accuracy = (correct_count/len(y_test.values)) * 100

print("Prediction Accuracy: ", round(accuracy, 2), "%")

# ******************************
# TRYING REGRESSION
# ******************************

regression_labels = {"Linear Regression": (LinearRegression(), "red"),
                     "ElasticNet": (ElasticNet(precompute=True), "gold"),
                     "BayesianRidge": (BayesianRidge(), "purple"),
                     "KernelRidge - Linear": (KernelRidge(kernel='linear'), "blue"),
                     "KernelRidge - Poly": (KernelRidge(kernel='poly'), "darkorange"),
                     "KernelRidge - RBF": (KernelRidge(kernel='rbf'), "green")}

plt.clf()
df_final.drop(['registered', 'casual'], inplace=True, axis=1)
fig4 = plot_regression_curves(reg_arr=regression_labels, X=x_lda_noreg, y=df_count.values,
                   title="Regression Cross Validation Accuracy on Yulu Dataset with LDA")
plt.savefig("regression_accuracy")
plt.show()

# Try regression with Linear Regression
x_train, x_test, y_train, y_test = train_test_split(x_lda_noreg, df_count.values, train_size=0.8, random_state=1, shuffle=True)

r_rbf = KernelRidge(kernel='rbf')
r_rbf.fit(x_train, y_train)
# Get predicted values to calculate RMSE and MSE
y_pred = r_rbf.predict(x_test)

print("\n\nTraining Results Regression")
print("-----------------------------------------------------")
print("Coefficient of Determination:", round(r2_score(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))


correct_count = 0
y_pred = np.round(y_pred).astype(int)
for i in range(len(y_test)):
    if (y_test[i] >= (y_pred[i] - 25) and y_test[i] <= (y_pred[i]  + 25)):
        correct_count += 1

# Calculate accuracy of predictions
accuracy = (correct_count/len(y_test)) * 100
print("\nRegression Prediction Accuracy: ", round(accuracy, 2), "%")




