from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class KNN:


    def __init__(self, k_nghbs, n_ths, train, valid, train_l, valid_l):
        
        self.knn = KNeighborsClassifier(n_neighbors=k_nghbs, n_jobs=n_ths)

        self.train = train
        self.train_l = train_l
        self.valid = valid
        self.valid_l = valid_l


    def get_report(self):
        report = dict()

        prediction = self.knn.predict(self.train)
        report.update( {'prediction' : prediction} )

        classification = classification_report(self.valid_l, prediction)
        report.update( {'classification' : classification} )

        conf_matrix = confusion_matrix(self.valid, prediction)
        report.update( {'confusion matrix' : conf_matrix} )

        accuracy = self.knn.score(self.valid, self.valid_l)
        report.update( {'accuracy' : accuracy} )

        return report


    def train(self):

        # To teach a model via Training group
        self.knn.fit(self.train, self.train_l)


    def get_knn(self):
        return self.knn

