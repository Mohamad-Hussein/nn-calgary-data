import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split


df = pd.read_csv('reduced_version_data_ENEL_645.csv')

# showing features
listing = [print(x) for x in df.columns]
print(f"\nLenght of features in dataset: {len(df.columns)}")
print(df.head())

# sorting df by community name and year
sorted_df = df.sort_values(by=["Community Name","Year"])

data_len = sorted_df.size

# Going to check how many items for each community
value_counts = sorted_df["Community Name"].value_counts()

# grouping by community name
grouped = df.groupby("Community Name")


"""These functions were made to debug the data"""
# -------------------------------------------------

def find_zero_columns(X_df):
    x = X_df.to_numpy()
    zero_columns = np.all(x == 0, axis=0)
    # print(zero_columns)

    # np.set_printoptions(threshold=np.inf)
    index_list_zero_col = []
    for index,element in enumerate(zero_columns):
        if element == True:
            index_list_zero_col.append(index)
    # print(index_list_zero_col)
    col_list = X_df.columns.values.tolist()
    for index in index_list_zero_col:
        print(col_list[index])
        print(X_df[col_list[index]])

def mismatching_cols(X_train, X_test):
    """
    This function is to find the mismatching columns between the training and testing data
    """
    mismatching_cols_list = []
    index_list = []
    for index,feature in enumerate(X_train.columns):
        if feature not in X_test.columns:
            mismatching_cols_list.append(feature)
            index_list.append(index)
    print(f"\nColumns: {X_train.columns}")
# -------------------------------------------------


def calc_coefficients(X_df, Y_df):
    """
    This function is to calculate the coefficients of the linear regression"""
    

    x = X_df.to_numpy()
    x = np.insert(x, 0, 1, axis=1)
    print(x)
    y = Y_df.to_numpy()

    
    # problem where original training data contains full row of 0s
    # I need to clear the rows of 0's in the training data
    find_zero_columns(X_df)

    B = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

    return B

def test(coeff, X_df, Y_df):
    X_test = X_df.to_numpy()
    X_test = np.insert(X_test, 0, 1, axis=1)
    Y_test = Y_df.to_numpy()
    pred = X_test @ coeff
    pred = np.round(pred)

    print(f"Predicted: {pred}")
    print(f"Actual: {Y_test}")

    total_correct = 0
    for x,y in zip(pred,Y_test):
        if x == y:
            total_correct += 1
    accuracy = total_correct / len(Y_test)

    print(f"\nAccuracy: {accuracy * 100}%")
    print(f"Total correct: {total_correct}")
    print(f"MSE: {round(np.mean((pred-Y_test) ** 2))}")

def data_cleaning(df):
    # This caused problems because there is only one row of this community
    filtered_df = df.drop(df[df["Community Name"] == "05F"].index)
    # This is to remove the instance of mistake in the data, a category with only count of 1
    unique = filtered_df["Category"].value_counts()
    unique_ilist = unique[unique == 1].index.tolist()
    filtered_df.drop(filtered_df[filtered_df['Category'].isin(unique_ilist)].index, axis=0, inplace=True)
    return filtered_df

def main():
    # Starting the calculation process

    # I need to remove the instances with a community name mentionned only once,
    # this is because the split will not work if there is only one instance of a community
    
    
    # Now theres a new problem that when dummmies are made, train has one more feature than test
    # mismatching_cols(X_train, X_test)

    # THIS IS THE PROBLEM, YOU MADE THE DUMMIES AND THE SPLIT MADE SOME COMMUNITIES MISSING
    # THUS THE COLLUMN OF ZEROES

    filtered_df = data_cleaning(df)

    X = filtered_df.drop("Crime Count", axis=1)
    Y = filtered_df["Crime Count"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=X['Community Name'],test_size=0.3, random_state=42)
    X_train= pd.get_dummies(X_train,dtype=float)
    X_test= pd.get_dummies(X_test,dtype=float)

    coefficients = calc_coefficients(X_train, y_train)
    print(f"\n\nThese are the coefficients: {coefficients}\n\n")

    test(coefficients, X_test, y_test)
    
    
    return 0

if __name__ == '__main__':
    main()