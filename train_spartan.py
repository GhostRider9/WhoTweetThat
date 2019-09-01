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
    test_set = []
    training_classifications = []
    for _, row in train_norm.iterrows():
        feature_dict = feature_extractor(row["Twitter"].split("\x01"))
        training_set.append(feature_dict)
        training_classifications.append(row["UID"])
    for _, row in test_norm.iterrows():
        features = feature_extractor(row["Twitter"].split("\x01"))
        test_set.append(features)
    vectorizer = DictVectorizer()
    training_data = vectorizer.fit_transform(training_set)
    test_data = vectorizer.transform(test_set)
    return training_data, training_classifications, test_data


# save predicted result to csv file for uploading
def save_predicted(predicted, index):
    output = [(i + 1, pred) for i, pred in enumerate(predicted)]
    out_df = pd.DataFrame(output, columns=["Id", "Predicted"]).set_index("Id")
    out_df.to_csv(data_folder + "predicted_{}.csv".format(index))


# fit selected models and predict result
def fit_predict(clfs, indexs, data, classifications):
    for i in indexs:
        s_time = time()
        clfs[i].fit(data, classifications)
        save_predicted(clfs.predict(test_data), i)
        with open('trained_{}.pkl'.format(i), 'wb') as fid:
            pickle.dump(clfs[i], fid, protocol=4)
        print("time cost:{}".format(time() - s_time))


# load existing models and predict result
def load_predict(indexs):
    for i in indexs:
        with open('trained_{}.pkl'.format(i), 'rb') as fid:
            model = pickle.load(fid)
        save_predicted(model.predict(test_data), i)


data_folder = "/home/zlp/SML/"
train_norm = pd.read_csv(data_folder + "train_norm.csv")
test_norm = pd.read_csv(data_folder + "test_norm.csv")

trn_data, trn_classes, test_data = prepare_data(get_BOW)

clfs = [KNeighborsClassifier(n_jobs=-1), DecisionTreeClassifier(), RandomForestClassifier(n_jobs=-1),
        MultinomialNB(), LinearSVC(), LogisticRegression(n_jobs=-1)]

models_index = list(range(2, 3))  # select a list of models
# fit_predict(clfs, models_index, trn_data, trn_classes)
load_predict(models_index)
