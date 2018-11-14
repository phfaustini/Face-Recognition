import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')


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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

        n_components = 100
        pca = PCA(n_components=n_components,random_state=42, whiten=True).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        print("SVM: ")
        clf = SVC(random_state=42)
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        print(classification_report(y_test, y_pred, target_names=target_names, labels=np.array(target_names)))
        print()

        print("LDA: ")
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        print(classification_report(y_test, y_pred, target_names=target_names, labels=np.array(target_names)))
        print()

        print("Naive Bayes: ")
        clf = GaussianNB()
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        print(classification_report(y_test, y_pred, target_names=target_names, labels=np.array(target_names)))
        print()
