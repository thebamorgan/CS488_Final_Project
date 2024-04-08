import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error

# Read in data
df = pd.read_csv("bike_rental_data.csv")

# Convert date time into a structure where it is easier to extract data
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df.drop('datetime', axis=1, inplace=True)

# Drop unnecessary columns
df.drop(['season', 'holiday', 'workingday', 'weather'], axis=1, inplace=True)

X = df[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'hour']]
y = df['count']

test_sizes = [0.2, 0.4, 0.6, 0.8]  # Test sizes to iterate over

for test_size in test_sizes:
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initializing the Naive Bayes classifier
    nb_classifier = GaussianNB()

    # Training the classifier
    nb_classifier.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = nb_classifier.predict(X_test)

    # Evaluating the model
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test Size: {test_size}, Mean Absolute Error: {mae}")

    # Visualizing the predicted versus actual counts
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Count')
    plt.ylabel('Predicted Count')
    plt.title(f'Actual vs. Predicted Counts (Test Size: {test_size})')
    plt.show()
