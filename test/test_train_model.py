import pandas as pd
import random
from numpy import ndarray

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing._encoders import OneHotEncoder
from application.train_model import process_data, train_model, predict

# Creating DF
# A list of 10 features
num_users=1000
features = [
    "gender",
    "rating"
]
df = pd.DataFrame(columns=features)
genders = ["male", "female", "na"]
df['gender'] = random.choices(
    genders,
    weights=(47,47,6),
    k=num_users
)
# The different ratings available
ratings = ["1","2","3","4","5"]# Weighted ratings with a skew towards the ends
df['rating'] = random.choices(
    ratings,
    weights=(30,10,10,10,30),
    k=num_users
)
print(df.dtypes)
print(df)



def test_process_data():
    X_train, y_train, encoder, lb = process_data(df, ['gender'], 'rating')
    assert isinstance(X_train, pd.core.frame.DataFrame)
    assert isinstance(y_train, pd.core.frame.Series)
    assert isinstance(encoder, OneHotEncoder)

def test_train_model():
    X_train, y_train, _, _ = process_data(df, ['gender'], 'rating')
    lr_model = train_model(X_train, y_train)
    assert isinstance(lr_model, LogisticRegression)

def test_predict():
    X_train, y_train, _, _ = process_data(df, ['gender'], 'rating')
    lr_model = train_model(X_train, y_train)
    y_pred = predict(lr_model, X_train)
    assert isinstance(y_pred, ndarray)