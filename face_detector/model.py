import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')


class Model():
    
    def _classification_report_with_accuracy_score(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))
        return accuracy_score(y_true, y_pred)
    
    def _show_results(self, clf, data, y):
        nested_score = cross_val_score(clf, data, y, cv=StratifiedKFold(3, random_state=1), scoring=make_scorer(self._classification_report_with_accuracy_score))
        print("Accuracy in each run = {0}".format(nested_score))
        print("Accuracy (general): %0.2f (+/- %0.2f)" % (nested_score.mean(), nested_score.std() * 2))
        print("------------------------------------------------------------")
        print()

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

        pca = PCA(n_components=50, random_state=42, whiten=True).fit(X)
        X_train_pca = pca.transform(X)

        print("SVM: ")
        clf = SVC(random_state=42)
        self._show_results(clf, X_train_pca, y)

        print("LDA: ")
        clf = LinearDiscriminantAnalysis()
        self._show_results(clf, X_train_pca, y)

        print("Naive Bayes: ")
        clf = GaussianNB()
        self._show_results(clf, X_train_pca, y)

        print("3NN: ")
        clf = KNeighborsClassifier(n_neighbors=3)
        self._show_results(clf, X_train_pca, y)
