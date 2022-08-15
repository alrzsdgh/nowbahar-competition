# Importing needed modules
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# Importing dataset
train_df = pd.read_csv('../input/school2/X_train.csv')
test_df = pd.read_csv('../input/school2/X_test.csv')
label_df = pd.read_csv('../input/school2/y_train.csv')

# Separating categorical and non-categorical features
cat_index = []
for k,v in enumerate(train_df.dtypes):
    if v == object:
        cat_index.append(k)

df_indices = []
for i in train_df:
    df_indices.append(i)

cat_lst = []
for i in cat_index:
    cat_lst.append(df_indices[i])

uncat_lst = []
for i in range(1,33):
    if i in cat_index:
        continue
    uncat_lst.append(df_indices[i])

# Data preprocessing pipeline
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, uncat_lst),
    ('cat', OneHotEncoder(), cat_lst),
])

x_train = full_pipeline.fit_transform(train_df)

y_train = label_df['G3'].values
y_train = np.expand_dims(y_train, axis = -1)


# Random forest model
forest_reg = RandomForestRegressor(n_estimators=10000)
forest_reg.fit(x_train,y_train)
forest_pred = forest_reg.predict(x_train)
forest_mse = mean_squared_error(y_train, forest_pred)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
scores = cross_val_score(forest_reg, x_train, y_train, scoring='neg_mean_squared_error',cv=10)
f_rmse_scores = np.sqrt(-scores)

scores.mean()

# Predicting values in test dataset
x_test = full_pipeline.fit_transform(test_df)

pred = forest_reg.predict(x_test)

df = pd.DataFrame()
df['StudentID'] = test_df['StudentID']
df['Score'] = pred

df.to_csv('./sub.csv')
