#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


 
def data_preprocessing(df):
    # This is to remove the instance of mistake in the data, a category with only count of 1
    # kind of useless now because there is no item with a category of count 1
    filtered_df = df
    unique = filtered_df["Category"].value_counts()
    unique_ilist = unique[unique == 1].index.tolist()
    filtered_df.drop(filtered_df[filtered_df['Category'].isin(unique_ilist)].index, axis=0, inplace=True)
    return filtered_df

# %%
#%%
data_path = os.path.join('..', 'reduced_version_data_ENEL_645.csv')
df = pd.read_csv(data_path)
filtered_df = data_preprocessing(df)
X = filtered_df.drop(["Category"], axis=1)
y = filtered_df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,test_size=0.3,random_state=42)

X_train = pd.get_dummies(X_train, dtype=int)
X_test = pd.get_dummies(X_test, dtype=int)

## Scaling continuous features so it could learn faster
scaler = StandardScaler()

col_to_scale = ['Crime Count','Resident Count','Year']
X_train[col_to_scale] = scaler.fit_transform(X_train[col_to_scale])

X_test[col_to_scale] = scaler.fit_transform(X_test[col_to_scale])

# %%
#%%
## Features to remove
train_cols = X_train.columns
test_cols = X_test.columns
col_rm = list(set(train_cols).difference(test_cols))
X_train.drop(col_rm,axis=1,inplace=True)

col_add = list(set(test_cols).difference(train_cols))
index = test_cols.get_loc(col_add[0])
X_train.insert(index,col_add[0],0)

# %%
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)

# %%
accuracy = model.score(X_test, y_test)
accuracy

# %%
model.predict(X_test)

# %%
y_test.to_numpy()

# %%



