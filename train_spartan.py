import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction import DictVectorizer
import pickle
from time import time


def get_BOW(text):
    BOW = {}
    for word in text:
        BOW[word] = BOW.get(word, 0) + 1
    return BOW


def prepare_data(feature_extractor):
    training_set = []
    training_classifications = []
    for _, row in train_norm.iterrows():
        feature_dict = feature_extractor(row["Twitter"].split("\x01"))
        training_set.append(feature_dict)
        training_classifications.append(row["UID"])

    vectorizer = DictVectorizer()
    training_data = vectorizer.fit_transform(training_set)
    return training_data, training_classifications


def fit_models(clfs, data, classifications):
    index = 0
    for clf in clfs:
        s_time = time()
        clf.fit(data, classifications)
        with open('trained_{}.pkl'.format(index), 'wb') as fid:
            pickle.dump(clf, fid)
        index += 1
        print("time cost:{}".format(time() - s_time))


data_folder = "/home/zlp/SML/"
train_norm = pd.read_csv(data_folder + "train_norm.csv")
test_norm = pd.read_csv(data_folder + "test_norm.csv")

trn_data, trn_classes = prepare_data(get_BOW)

clfs = [KNeighborsClassifier(n_jobs=-1), DecisionTreeClassifier(), RandomForestClassifier(n_jobs=-1),
        MultinomialNB(), LinearSVC(), LogisticRegression(n_jobs=-1)]

fit_models(clfs, trn_data, trn_classes)
