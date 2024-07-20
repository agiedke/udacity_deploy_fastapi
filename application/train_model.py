# Script to train machine learning model.
import sys
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Constants
LOCALPATHTODATA = "./nd0821-c3-starter-code/starter/data/census.csv"
MODELPATH="model/lr_model.pkl"

# Functions
# Proces the test data with the process_data function.
def process_data(train, categorical_features, label, training=True, encoder=None):
    # categorical features
    X_train_cat_raw = train[categorical_features]
    if training:
        # one hot encoding
        encoder = OneHotEncoder()
        encoder.fit(X_train_cat_raw)
    if not training and encoder is None:
        print("Error: Need to pass encoder")
        sys.exit(1)
    onehotlabels = encoder.transform(X_train_cat_raw).toarray()
    # converting to dataframe
    new_columns = list()
    for col, values in zip(X_train_cat_raw.columns, encoder.categories_):
        new_columns.extend([col + '_' + str(value) for value in values])
    X_train_cat = pd.DataFrame(onehotlabels, columns=new_columns)

    # numerical features
    X_train_num = train.select_dtypes(exclude='object')

    # all features
    X_train = pd.concat([X_train_cat.reset_index(drop=True), X_train_num.reset_index(drop=True)], axis=1)

    y_train = train[label].astype('category')

    lb = new_columns
    return X_train, y_train, encoder, lb

def train_model(X_train, y_train, model_path):
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    print(type(lr_model))
    pickle.dump(lr_model, open(model_path, "wb"))
    return lr_model

def predict(model, X):
    return model.predict(X)

def score(y_pred, y_true):
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    print(f"Overall score f1 macro: {f1_macro}")
    print(f"Overall score f1 micro: {f1_micro}\n")

def sliced_score(X_test, y_test, model, lb, min_sample_size=30):
    y_test_tmp = y_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    nr_cat = 0
    for _cat in lb:
        x_test_sub = X_test[X_test[_cat]==1]
        sample_size = len(x_test_sub)
        if sample_size>=min_sample_size:
            print(f"category: {_cat}")
            y_pred_sub = pd.Series(model.predict(x_test_sub))
            y_test_sub = y_test_tmp[y_test_tmp.index.isin(list(y_pred_sub.index))]
            print(f"sample size : {len(y_test_sub)}")
            f1_macro = f1_score(y_test_sub, y_pred_sub, average='macro')
            f1_micro = f1_score(y_test_sub, y_pred_sub, average='micro')
            print(f"f1 macro: {f1_macro}")
            print(f"f1 micro: {f1_micro}\n")
            nr_cat+=1
    print(f"Number of categories with sample size over {min_sample_size}: {nr_cat}")


if __name__=="__main__":
    # Add code to load in the data.
    data = pd.read_csv(LOCALPATHTODATA)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=999)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # process data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder
    )

    # Train and save a model.
    model = train_model(X_train, y_train, model_path=MODELPATH)

    # Get overall performance
    y_pred = predict(model, X_test)
    score(y_pred, y_test)

    # Get performance on slices
    #sliced_score(X_test, y_test, model, lb)
