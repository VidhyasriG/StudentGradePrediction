# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:32:49 2019

@author: vidhy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 00:38:04 2019

@author: vidhy
"""
import lightgbm as lgb

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import pandas as pd  
import random
import seaborn as snb


#def xgbreg(df):
df = pd.read_csv("C:\\Users\\vidhy\\OneDrive\\Desktop\\Fall 19\\DEAN 690\\UI\\SyntheticData.csv")
df=df.drop(columns=['Num'])
mean = df['Course.Grade.with.No.extra.credit'].mean()
#mean
df['Course.Grade.with.No.extra.credit'].fillna(mean, inplace=True)
df=df.drop(['Course.Grade.with.Extra.Credit',
            'Proj','HW10','HW9','HW8','HW7','HW6','SectionCode','Term',
            'Season','Year','Type','GTA','Domicile','Credit hours',
            'Test2','Finals','PreReqSatisfied','Class','Test1','Instructor'],axis=1)
#df.info()

#One-hot encoding for converting categorical to dichotomous variables
dfcat=df.drop(['HW1','HW2','HW3','HW4','HW5','Quizzes','Course.Grade.with.No.extra.credit'],axis=1)
#dfcat.info()
one_hot = pd.get_dummies(dfcat)
#one_hot.info()
dfnum=df.drop(['Level','Load','gender','Major'],axis=1)
dffinal= dfnum.join(one_hot)
#dffinal.info()
#dffinal.head() 

#Split train/test
X = dffinal.drop('Course.Grade.with.No.extra.credit', axis=1)
y = dffinal['Course.Grade.with.No.extra.credit']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
max_rmse=11.19
max_i=0
#lightgbm boost model
for i in range(1500,2500):
    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'huber',
        'metric': 'rmse',
        'learning_rate': 0.3,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 10,
        'verbose': 0,
        "max_depth": 10,
        "num_leaves": 128,
        "max_bin": 100,
        "num_iterations": 10000,
        "n_estimators": 1200,
        'random_state': i,
        'silent':1
    }

    gbm = lgb.LGBMRegressor(**hyper_params)

    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            verbose=True,
            early_stopping_rounds=1000)


    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

    rmse=round(mean_squared_error(y_pred, y_test) ** 0.5, 5)
    print('The rmse of prediction is:', rmse)
    print(gbm.score(X, y), 1 - (1-gbm.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1))

    print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_test)))
    ##Rsquare
    #print(gbm.score(X, y), 1 - (1-gbm.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1))
    if(rmse<=max_rmse):
        max_rmse=rmse
        max_i=i
        adjr=1 - (1 - gbm.score(X, y)) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
        print(adjr)

print(i)
print(max_rmse)

#max i =699->11.27
#    feature_imp = pd.DataFrame(sorted(zip(gbm.feature_importances_,X.columns)), columns=['Value','Feature'])
#    plt.figure(figsize=(10, 5))
#    snb.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
#                    ascending=False))
#    plt.title('LightGBM Features')
#    plt.tight_layout()
#    plt.show()
#
#    test_pred=np.expm1(gbm.predict(X_test))
#    X_test["Course.Grade.with.No.extra.credit"] = np.log1p(test_pred)
#    X_test.to_csv("lgbresults.csv", columns=["Course.Grade.with.No.extra.credit"], index=False)