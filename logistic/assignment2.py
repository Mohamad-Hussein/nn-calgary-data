from timeit import default_timer as timer
from numba import jit,cuda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split


 
def data_preprocessing(df):
    # This is to remove the instance of mistake in the data, a category with only count of 1
    # kind of useless now because there is no item with a category of count 1
    filtered_df = df
    unique = filtered_df["Category"].value_counts()
    unique_ilist = unique[unique == 1].index.tolist()
    filtered_df.drop(filtered_df[filtered_df['Category'].isin(unique_ilist)].index, axis=0, inplace=True)
    return filtered_df

# @jit(target_backend='cuda', nopython=True)
def classify(coefficients, data_array):
    ones_column = np.ones((data_array.shape[0], 1), dtype=float)
    data = np.hstack((ones_column, data_array))
    exponent = coefficients @ data.T
    return np.rint(1 / (1 + np.exp(-exponent)))

# @jit(target_backend='cuda', nopython=True)
def calc_error(answer, y_train, feature_index):
    total_wrong = 0
    total = y_train.shape[0]
    y = y_train[:,feature_index]
    for i in range(0, total):
        if answer[i] != y[i]:
            total_wrong += 1

    return total_wrong / total

def adjust_coeff(coeff, x, y, prev_error, prev_action):
    ans = classify(coeff, x)
    error = calc_error(ans, y)
            
def main():
    df = pd.read_csv('reduced_version_data_ENEL_645.csv')
    filtered_df = data_preprocessing(df)
    X = filtered_df.drop(["Category"], axis=1)
    y = filtered_df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,test_size=0.3,random_state=42)

    y_train = pd.get_dummies(y_train, dtype=int)
    y_test = pd.get_dummies(y_test, dtype=int)
    X_train = pd.get_dummies(X_train, dtype=int)
    X_test = pd.get_dummies(X_test, dtype=int)

    # coeff = cost_function(np.ones(X_train.shape[1] + 1), 0, X_train.to_numpy(), y_train.to_numpy())
    
    
    return 0

if __name__ == '__main__':
    main()