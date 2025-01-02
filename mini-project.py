from typing import Counter
import pandas as pd
import numpy as np
from scipy import linalg
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 

def load_dataset(dataset_name):
# Fetch dataset

    # Features (X) and Targets (y)
    X = dataset_name.data 
    y = dataset_name.target   
    
    return X, y


def preprocess_dataset(X, y):
    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# Implement k-NN Classifier
class KNearestNeighbors:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        best_k, _ = self.crossValidateKnn(self.X_train, self.y_train, self.k)
        print(f"Best k value selected by cross-validation: {best_k}")

        # return self.crossValidateKnn(self.X_train, X_test, self.k)
        return self.predictKNN(self.X_train, self.y_train, X_test, best_k)

    def predictKNN(self, Tr_set, Ltr_set, X, k):
        num_test = X.shape[0]
        Lpred = np.zeros(num_test, dtype=Ltr_set.dtype)

        for i in range(num_test):
            distances = np.sqrt(np.sum((Tr_set - X[i, :]) ** 2, axis=1))

            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = Ltr_set[nearest_indices]

            Lpred[i] = np.bincount(nearest_labels).argmax()

            tiedLabels = np.where(np.bincount(nearest_labels) == np.bincount(nearest_labels).max())[0]
            if len(tiedLabels) == 1:
                Lpred[i] = tiedLabels[0]
            else:
                # If there is a tie, select the label with the smallest distance
                tiedLabelsIndices = [index for index in nearest_indices if Ltr_set[index] in tiedLabels]
                closestIndex = tiedLabelsIndices[0]
                Lpred[i] = Ltr_set[closestIndex]

        return Lpred
    
    
    def crossValidateKnn(self,Tr_set, Ltr_set, k_values):
        numSamples=Tr_set.shape[0]
        foldSize=numSamples // 3
        accuraciesPerK={}

        foldX = [Tr_set[i*foldSize:(i+1)*foldSize] for i in range(3)]
        foldY = [Ltr_set[i*foldSize:(i+1)*foldSize] for i in range(3)]

        for k in k_values:
            accuracies=[]
            for i in range(3):

                xVal= foldX[i]
                yVal= foldY[i]

                xTrain=np.concatenate(foldX[:i]+foldX[i+1:])
                yTrain=np.concatenate(foldY[:i]+foldY[i+1:])

                yPred=self.predictKNN(xTrain,yTrain,xVal,k)
                
                accuracy = np.mean(yPred == yVal)
                accuracies.append(accuracy)

            accuraciesPerK[k]=np.mean(accuracies)

        bestK = max(accuraciesPerK, key=accuraciesPerK.get)
        return bestK, accuraciesPerK

class ESN():
    def __init__(self, input_dim, reservoir_size_Nx, output_dim, spectral_radius=0.8, input_scaling=0.2, seed=42):
        self.input_dim = input_dim
        self.reservoir_size_Nx = reservoir_size_Nx
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.last_training_state = None
        np.random.seed(seed)

        self.W_in = (np.random.uniform(-1, 1, [self.reservoir_size_Nx, self.input_dim + 1])) * self.input_scaling
        # Generate W
        self.W = np.random.uniform(-1, 1, [self.reservoir_size_Nx, self.reservoir_size_Nx])
        # Calculate the spectral radius of W (eigenvalues) and divide W by it
        eigenvalues = linalg.eigvals(self.W)
        max_abs_eigenvalue = max(abs(eigenvalues))
        print("Spectral radius of W: ", max_abs_eigenvalue)
        self.W = self.W / max_abs_eigenvalue
        # Scale the W matrix with "ultimate" spectral radius
        self.W = self.W * self.spectral_radius
        self.W_out = None
        
    def reservoir_update(self, u, x):
        u_bias = np.append(u, 1)
        new_state = np.tanh(np.dot(self.W_in, u_bias) + np.dot(self.W, x))
        return new_state

    def predictLeaf(self, X_test):
        predictions = []
        
        for i, signal in enumerate(X_test):
            # Initialize reservoir state to zero for each signal
            reservoir_state = np.zeros(self.reservoir_size_Nx)

            reservoir_state = self.reservoir_update(signal, reservoir_state)
            extended_state = np.hstack([reservoir_state, 1])
            output = extended_state @ self.W_out
            predictions.append(output)

        return np.array(predictions)


    def trainLeaf(self, X_train, y_train_onehot, reg_param):
        
        reservoir_states = []
        for signal in X_train:
            r_prev = np.zeros(self.reservoir_size_Nx)
            r_prev = self.reservoir_update(signal, r_prev)

            reservoir_states.append(r_prev)

        reservoir_states = np.array(reservoir_states)
        augmented_states = np.hstack([reservoir_states, np.ones((reservoir_states.shape[0], 1))])

        beta = reg_param
        I = np.eye(augmented_states.shape[1])
        self.W_out = np.linalg.solve(augmented_states.T @ augmented_states + beta * I, augmented_states.T @ y_train_onehot)



def iris():
    # Load dataset
    X, y = load_dataset(load_iris())
    X_train, X_test, y_train, y_test = preprocess_dataset(X, y)

    # Train and evaluate k-NN
    knn = KNearestNeighbors(k=[1,2,3,4,5])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    scaler = StandardScaler()
    X_train_ESN = scaler.fit_transform(X_train)
    X_test_ESN = scaler.transform(X_test)

    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_train_ESN = encoder.fit_transform(y_train.reshape(-1,1))

    reservoir_size = 1000
    spectral_radius = 0.99
    input_scaling = 0.25
    reg_param = 1e-8
    nr_of_simulations = 10    

    predictions = []

    for run in range(nr_of_simulations):
        esn = ESN(input_dim=X_train_ESN.shape[1], 
                reservoir_size_Nx=reservoir_size, 
                output_dim=y_train_ESN.shape[1], 
                spectral_radius=spectral_radius, 
                input_scaling=input_scaling,
                seed=run)

        esn.trainLeaf(X_train_ESN, y_train_ESN , reg_param=reg_param)

        prediction = esn.predictLeaf(X_test_ESN)
        predictions.append(prediction)

    print(f'{predictions=}')
    mean_predictions = np.mean(predictions, axis=0)

    # The predicted labels are of by one so we need to add 1 to the predicted labels
    predicted_labels = np.argmax(mean_predictions, axis=1) 
    print(f'{predicted_labels=}')
    print(f'{y_test=}')
    print(f'{accuracy_score(y_test, predicted_labels)=}')
# X,y = load_dataset(load_iris())
# X_train, X_test, y_train, y_test = preprocess_dataset(X, y)
# # Train and evaluate k-NN
# knn = KNearestNeighbors(k=3)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test) samuel var här 
# accuracy = np.mean(y_pred == y_test)
# print(f"Accuracy: {accuracy * 100:.2f}%")

def wine():
    # Load dataset
    X, y = load_dataset(load_wine())
    X_train, X_test, y_train, y_test = preprocess_dataset(X, y)

    # Train and evaluate k-NN
    knn = KNearestNeighbors(k=[1,2,3,4,5])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def breast_cancer():
    # Load dataset
    X, y = load_dataset(load_breast_cancer())
    X_train, X_test, y_train, y_test = preprocess_dataset(X, y)

    # Train and evaluate k-NN
    knn = KNearestNeighbors(k=[1,2,3,4,5])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def diabetes():
    # Load dataset
    X, y = load_dataset(load_diabetes())
    y = pd.qcut(y, q=3, labels=[0, 1, 2]).astype(int)
    X_train, X_test, y_train, y_test = preprocess_dataset(X, y)

    # Train and evaluate k-NN
    knn = KNearestNeighbors(k=[1,2,3,4,5])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def digits():
    # Load dataset
    X, y = load_dataset(load_digits())
    X_train, X_test, y_train, y_test = preprocess_dataset(X, y)

    # Train and evaluate k-NN
    knn = KNearestNeighbors(k=[1,2,3,4,5])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%") 

if __name__ == "__main__":
    iris()
    wine()
    breast_cancer()
    diabetes()
    digits()

