# Boilerplate Code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn import linear_model, preprocessing, dummy, model_selection, neighbors, metrics


def plot_data_predictions(training_data, prediction_data, prediction_results, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X1 = prediction_data.iloc[:, 0]
    X2 = prediction_data.iloc[:, 1]
    ax.scatter(X1, X2, list(prediction_results), color='#add8e680')
    X = training_data.iloc[:, 0]
    Y = training_data.iloc[:, 1]
    Z = training_data.iloc[:, 2]
    ax.scatter(X, Y, Z, c='#ff00ee', edgecolors='#ff0000', s=8)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('target', rotation=0)
    if title:
        plt.title(title)
    plt.show()


def processData(data):
    """
    Processes the data and returns a list of the processed data.
    """
    processedData = []
    for line in data:
        processedData.append(line.split(","))
        processedData[0] = line[2]  # max temp
        processedData[1] = line[4]  # min temp
        processedData[2] = line[8]  # rainfall

    return processedData


def dummyModel(data):
    print('Dummy Model')
    X1 = data.iloc[:,0]
    X2 = data.iloc[:,1]
    Y = data.iloc[:,2]
    X = np.stack((X1, X2))
    dummy_reg = dummy.DummyRegressor(strategy='mean').fit(X, Y)


def logisticRegression(data):
    print('Logistic Regression')
    X1 = data.iloc[:,0]
    X2 = data.iloc[:,1]
    Y = data.iloc[:,2]
    X = np.stack((X1, X2))
    poly_range = [1, 2, 3, 4, 5] # can add more just examples
    range_c = [0.01, 0.1, 1, 10] # again just examples
    for degree in poly_range:
        X_poly = preprocessing.PolynomialFeatures(degree).fit_transform(X)
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_poly, Y, test_size=0.2)
        for C in range_c:
            model = linear_model.LogisticRegression(C=C, penalty='l2', solver='lbfgs').fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            print('Degree =', degree)
            print('C =', C)
            plot_data_predictions(X_train, X_test, Y_pred)
            print('Confusion Matrix')
            print(metrics.confusion_matrix(Y_test, Y_pred))


def knn(data):
    print('Knn')
    X1 = data.iloc[:,0]
    X2 = data.iloc[:,1]
    Y = data.iloc[:,2]
    X = np.stack((X1, X2))
    k_range = [1, 5, 10, 20] #examples
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
    for k in k_range:
        model = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        print('k =', k)
        print('Confusion Matrix')
        print(metrics.confusion_matrix(Y_test, Y_pred))


if __name__ == "__main__":
    # Read the data from the file
    data = pd.read_csv('data/dublin-airport-1939/daily/dly532.csv', comment='#', header=25)
    # Process the data
    processedData = processData(data)
    # Print the processed data
    for line in processedData:
        print(line)
