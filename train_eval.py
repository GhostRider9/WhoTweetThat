import pandas as pd
import numpy as np
import pickle
import random
from collections import defaultdict


def read_data(path):
    train = pd.read_csv(path + "train_tweets.txt", sep="\t", header=None, names=["UID", "Twitter"])
    test = pd.read_fwf(path + "test_tweets_unlabeled.txt", header=None)
    print("Shape of training data:{}".format(train.shape))
    print("Shape of test data:{}".format(test.shape))
    return train, test


def train_model(df):
    pass


def predict():
    pass


def dummy(path, train, test):
    ids = train["UID"].unique().tolist()

    print("the number of user id:{}".format(len(ids)))
    output = []
    for index, row in test_df.iterrows():
        output.append([index+1, random.choice(ids)])
    out_df = pd.DataFrame(output, columns=["Id", "Predicted"]).set_index("Id")
    out_df.to_csv(path + "dummmy_out.csv")


def test():
    count = 0
    line_count = 0
    for line in open("/home/zlp/data/SML/test_tweets_unlabeled.txt"):
        if len(line) == 0:
            count += 1
        line_count += 1
    print("The number of empty lines:{}".format(count))
    print("The number of lines:{}".format(line_count))


if __name__ == "__main__":
    data_folder = "/home/zlp/data/SML/"
    train_df, test_df = read_data(data_folder)
    dummy(data_folder, train_df, test_df)
    test()
    # load_existing_model = 0
    # if load_existing_model:
    #     pickle.load(open(data_folder + "pickled.model"))
    # else:
    #     model = train_model(train_df)
    #     pickle.dump(model, data_folder + "pickled.model")
