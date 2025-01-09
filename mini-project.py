import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder

# sunlar-3@student.ltu.se, lelrek-1@student.ltu.se

def fetch_ucirepo_data(id): 
    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features.to_numpy()  
    y = dataset.data.targets.to_numpy().ravel()
    return X, y

# Function to preprocess dataset
def preprocess_dataset(X, y):
    # Lable target data
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # Split the data into train, validate, test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    # Scale the data acording to the train data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test

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


class KMeansClassifier:
    def __init__(self, n_clusters=3, n_splits=5, random_state=42):
        self.n_clusters = n_clusters
        self.n_splits = n_splits
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def fit(self, X):
        self.kmeans.fit(X)
        return self.kmeans.labels_

    def predict(self, X):
        return self.kmeans.predict(X)

    def map_labels(self, y_true, y_pred):
        labels = np.zeros_like(y_pred)
        for i in range(self.n_clusters):
            mask = (y_pred == i)
            if np.any(mask):
                bincounts = np.bincount(y_true[mask])
                # Assign the most common label
                labels[mask] = np.argmax(bincounts)
        return labels

    def cross_validate(self, X, y, k_values):
        best_k = None
        best_score = -1
        all_scores = {}

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            silhouette_scores = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                kmeans.fit(X_train)
                y_pred = kmeans.predict(X_test)
                mapped_labels = self.map_labels(y_test, y_pred)
                silhouette_scores.append(silhouette_score(X_test, y_pred))

            avg_score = np.mean(silhouette_scores)
            all_scores[k] = avg_score

            if avg_score > best_score:
                best_score = avg_score
                best_k = k

        return best_k, all_scores
    
    def train_and_test(self, name, X_train, X_validation,X_test, y_train, y_validation, y_test):
        y_train_pred = self.kmeans.fit_predict(X_train)
        y_validation_pred = self.kmeans.predict(X_validation)
        y_test_pred = self.kmeans.predict(X_test)


        y_train_mapped = self.map_labels(y_train, y_train_pred)
        y_validation_mapped = self.map_labels(y_validation, y_validation_pred)
        y_test_mapped = self.map_labels(y_test, y_test_pred)

        train_accuracy = accuracy_score(y_train, y_train_mapped)
        validation_accuracy = accuracy_score(y_validation, y_validation_mapped)
        test_accuracy = accuracy_score(y_test, y_test_mapped)

        print(f"{name} - Validation Accuracy: {validation_accuracy * 100:.2f}%")
        print(f"{name} - Test Accuracy: {test_accuracy * 100:.2f}%")

def run_kMeans(name, X_train, X_validation,X_test, y_train, y_validation, y_test, k):

    kmeans_classifier = KMeansClassifier()
    best_k, scores = kmeans_classifier.cross_validate(X_train, y_train, k)
    print(f"Finished training k-Means for {name}")
    print(f"Best k value selected by cross-validation: {best_k}")
    print(f'{scores=}')
    kmeans_classifier.n_clusters = best_k
    kmeans_classifier.train_and_test(name, X_train, X_validation,X_test, y_train, y_validation, y_test)


def run_knn(name, X_train, X_validation,X_test, y_train, y_validation, y_test, k):
    # Train and evaluate k-NN
    knn = KNearestNeighbors(k=k)
    knn.fit(X_train, y_train)
    y_validation_pred = knn.predict(X_validation)
    y_pred = knn.predict(X_test)
    accuracy_validation = np.mean(y_validation_pred == y_validation)
    accuracy = np.mean(y_pred == y_test)
    print(f"Finished training k-NN for {name}")
    print(f"Accuracy for {name} (Validation): {accuracy_validation * 100:.2f}%")
    print(f"Accuracy for {name}: {accuracy * 100:.2f}%")
    


def run_experiment_ucirepo(name, id ,k=3, preprocess_callback=None):
    X, y = fetch_ucirepo_data(id)
    # Handle optional preprocessing callback (e.g., for diabetes)
    if preprocess_callback:
        y = preprocess_callback(y)
    X_train, X_validation, X_test, y_train, y_validation, y_test= preprocess_dataset(X, y)

    run_knn(name, X_train, X_validation, X_test, y_train, y_validation, y_test, k)
    run_kMeans(name, X_train, X_validation, X_test, y_train, y_validation, y_test, k)



if __name__ == "__main__":

    k = range(2,10)

    run_experiment_ucirepo("Iris", 53, k)
    run_experiment_ucirepo("Wine", 109, k)
    run_experiment_ucirepo("Breast Cancer", 17, k)
    run_experiment_ucirepo("Digits",80, k)
    run_experiment_ucirepo("Heart Failure Clinical Records", 519, k)
    run_experiment_ucirepo("Wine Qualitity", 186, k)
