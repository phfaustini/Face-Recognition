import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class Model():

    def classify(self, data: list):
        """
        :param data: [(img: np.ndarray, label: str)]
        """
        X = list(map(lambda t: t[0], data))
        y = list(map(lambda t: t[1], data))
        X = np.array(X)
        target_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        y = np.array(y)
        nsamples, nx, ny = X.shape
        X = X.reshape((nsamples, nx*ny))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        #  https://pythonmachinelearning.pro/face-recognition-with-eigenfaces/
        n_components = 100
        pca = PCA(n_components=n_components, whiten=True).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        clf = SVC()
        clf.fit(X_train_pca, y_train)
        #clf = MLPClassifier(hidden_layer_sizes=(1024,), solver='lbfgs', batch_size=256).fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        print(classification_report(y_test, y_pred, target_names=target_names))
