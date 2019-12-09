# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:58:42 2019

@author: vidhy
"""

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import pandas as pd  
import random


def train_predict(train_df, test_df):

    #  Train the model
    #  train_df=pd.read_csv("C:\\Users\\vidhy\\OneDrive\\Desktop\\Fall 19\\DEAN 690\\UI\\SyntheticData.csv")
    #  Reading & replacing missing values with mean for the target variable train_df
    mean = train_df['Course.Grade.with.No.extra.credit'].mean()
    train_df['Course.Grade.with.No.extra.credit'].fillna(mean, inplace=True)

    #  One-hot encoding for converting categorical to dichotomous variables
    df_cat = train_df[['Level', 'Load', 'gender', 'Major']]
    one_hot = pd.get_dummies(df_cat)
    df_num = train_df[['HW1', 'HW2', 'HW3', 'HW4', 'HW5', 'Quizzes', 'Course.Grade.with.No.extra.credit',
                       'Credit hours']]
    df_final = df_num.join(one_hot)

    #  Split for training
    x = df_final.drop(['Course.Grade.with.No.extra.credit'], axis=1)
    y = df_final['Course.Grade.with.No.extra.credit']

    # train_test_split to identify the best params (80/20)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1111)

    #  fitting the model with training data
    model = xgb.XGBRegressor()
    model.fit(x_train, y_train)

    #  Grid search

#     parameters = { 'nthread':[4], #when use hyperthread, xgboost may become slower
#                 'learning_rate': [0.01, .03, 0.05,0.07,0.09], #so called `eta` value
#                 'max_depth': [1,2,3,4,5],
#                 # 'gamma': [i/10.0 for i in range(0,5)],
#                 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
#                 'min_child_weight': [1,4,8,10],
#                 'silent': [1],
#                 'subsample': [0.6,0.7,0.8,1.0],
#                 'colsample_bytree': [0.6,0.7,0.8,1.0],
#                 'n_estimators': [1000]}
    # xgb_grid = GridSearchCV(model, parameters, cv=2, n_jobs=5, verbose=True, scoring='neg_mean_squared_error')
    # xgb_grid.fit(x, y)
    #    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bynode=1, colsample_bytree=0.6, gamma=0,
#       importance_type='gain', learning_rate=0.01, max_delta_step=0,
#       max_depth=2, min_child_weight=4, missing=None, n_estimators=1000,
#       n_jobs=1, nthread=4, objective='reg:linear', random_state=1111,
#       reg_alpha=0.01, reg_lambda=1, scale_pos_weight=1, seed=None,
#       silent=1, subsample=0.7, verbosity=1)
    
    #  Fit the model with the best parameters
    # model = xgb_grid.best_estimator_
    # n_est=80, learning_rate=0.04, colsample_bytree=0.7, random_state=11
    model=xgb.XGBRegressor(base_score=0.5, booster='dart', colsample_bylevel=1,
                           colsample_bynode=1, colsample_bytree=0.8, gamma=0,
                           importance_type='gain', learning_rate=0.04, max_delta_step=0,
                           max_depth=3, min_child_weight=1, missing=None, n_estimators=80,
                           n_jobs=1, nthread=None, random_state=11,
                           reg_alpha=0.01, reg_lambda=1, scale_pos_weight=1, seed=None,
                           silent=1, subsample=0.7, verbosity=1)
    model.fit(x_train, y_train)

    #  Cross validation with k=10 to train the model
    k_fold = KFold(n_splits=10, shuffle=True)
    kf_cv_scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring='neg_mean_squared_error')
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

    #  check the accuracy. We'll use MAE and RMSE as accuracy metrics.
    #  predict the grades for X_test data

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print("RMSE: %.2f" % np.sqrt(mse))
    mae = (mean_absolute_error(y_pred, y_test))
    print("Mean Absolute Error : " + str(mae))

    # #########################################################################################################
    #  Predict the grades for the test data
    # test_df=pd.read_csv("C:\\Users\\vidhy\\OneDrive\\Desktop\\Fall 19\\DEAN 690\\UI\\testdf.csv")
    #  One-hot encoding for converting categorical to dichotomous variables
    df_cat = test_df[['Level', 'Load', 'gender', 'Major']]
    one_hot = pd.get_dummies(df_cat)
    df_num = test_df[['HW1', 'HW2', 'HW3', 'HW4', 'HW5', 'Quizzes', 'Course.Grade.with.No.extra.credit',
                       'Credit hours']]
    df_final = df_num.join(one_hot)

    #  separate the target variable
    test_df_x = df_final.drop(['Course.Grade.with.No.extra.credit'], axis=1)
    test_df_y = df_final['Course.Grade.with.No.extra.credit']
    # print(df_final.columns)
    #  predict the grades for X_test data

    test_df_y_pred = model.predict(test_df_x)
    mse = mean_squared_error(test_df_y, test_df_y_pred)
    print("Test RMSE: %.2f" % np.sqrt(mse))
    mae = (mean_absolute_error(test_df_y_pred, test_df_y))
    print("Test Mean Absolute Error : " + str(mae))

    #  plot feature importance
    xgb.plot_importance(model)

    #  store results in csv
    #  test_pred = np.expm1(model.predict(X_test))
    results=pd.DataFrame()
    grade = model.predict(test_df_x)
    
    #grade = np.log1p(grade)
    results['Student_Num']=test_df['Num']
    results['Grades']=grade
    results.Grades=results.Grades.round(2)
    results.to_csv("xgbresults.csv", index=False)
    grades = pd.read_csv("xgbresults.csv")
    print(grades.columns)
    print("Grades predicted successfully!")
    return grades

################################################################################################

#  check the model with cross validation with 5 fold
#  scores = cross_val_score(model,X_train,y_train, cv=5)
#  print("Mean cross-validation score: %.2f" % scores.mean())

#  Cross-validation with a k-fold method
#  kfold = KFold(n_splits=10, shuffle=True)
#  kf_cv_scores = cross_val_score(model,X_train,y_train, cv=kfold, scoring='mean_squared_error' )
# print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

#  #predict test data and check its accuracy. We'll use MSE and RMSE as accuracy metrics.
#  ypred = model.predict(X_test)
#  mse = mean_squared_error(y_test,ypred)
#  print("MSE: %.2f" % mse)
#  print("RMSE: %.2f" % np.sqrt(mse))
#  print("Mean Absolute Error : " + str(mean_absolute_error(ypred, y_test)))
#  print("R square :")
#  print(model.score(X, y), 1 - (1-model.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1))

#  #visualize the original and predicted test data in a plot
#  x_ax = range(len(y_test))
#  plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
#  plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
#  plt.legend()
#  plt.show()
#  print(plt)

#  xgb.plot_importance(model)
#  importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
#  for  f in importance_types:
#     impPlot=model.get_booster().get_score(importance_type=f)
#     plt.title("Importance by " + f)
#     plt.xlabel("Relative Importance")
#     plt.ylabel("Features")
#     plt.barh(range(len(impPlot)),list(impPlot.values()))
#     plt.yticks(range(len(impPlot)),list(impPlot.keys()))
#     plt.show()

#  test_pred=np.expm1(model.predict(X_test))
#  X_test["Course.Grade.with.No.extra.credit"] = np.log1p(test_pred)
#  X_test.to_csv("xgbresults.csv", columns=["Course.Grade.with.No.extra.credit"], index=False)
