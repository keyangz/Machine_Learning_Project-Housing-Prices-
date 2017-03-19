import numpy as np
import pandas as pd
from feature_engineering import transform_features, transform_target, create_interaction_terms
from utils import optimize_and_evaluate_model
import csv
import operator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

########### Data Pre-Processing #############

# Read data
file_path = '../data/'
df_train = pd.read_csv(file_path + 'train.csv')
df_test = pd.read_csv(file_path + 'test.csv')

# Remove huge basement outlier from training data
df_train = df_train.drop(df_train[df_train['TotalBsmtSF'] > 4000].index)

# Split train into x and y
df_train_y = df_train.loc[:, ['SalePrice']]
df_train_x = df_train.drop(['SalePrice'], 1)

# Feature Engineering
df_train_y = transform_target(df_train_y) # train y
df_train_x = transform_features(df_train_x) # train x
df_test_x = transform_features(df_test) # test x

# Encode Categorical Variables and put into sci-kit friendly format
df_train_x = pd.get_dummies(df_train_x, drop_first=False, dummy_na=False)
df_train_x = df_train_x.fillna(0)
df_test_x = pd.get_dummies(df_test_x, drop_first=False, dummy_na=False)
df_test_x = df_test_x.fillna(0)

# Ensure test and training have the same variables
test_cols = set(df_test_x.columns.values)
train_cols = set(df_train_x.columns.values)
# Remove cols from train that are not in test
df_train_x = df_train_x.drop(list(train_cols-test_cols),1)
# Remove cols from train that are not in test
df_test_x = df_test_x.drop(list(test_cols-train_cols),1)

# Double check that all columns are the same in test and train
for test_col, train_col in zip(list(df_test_x.columns.values),list(df_train_x.columns.values)):
    if test_col != train_col:
        print test_col, train_col, 'not the same'

# Check for NaN
# print(pd.isnull(df_train_x).sum() > 0)

# Convert to matrix
train_x = df_train_x.as_matrix()
train_y = df_train_y.as_matrix().ravel()
test_x = df_test_x.as_matrix()


########### Lasso Regression #############

# Define model pipeline
pipeline_lasso = Pipeline(steps=[('lasso',Lasso(alpha=0.0001, normalize=True, max_iter=100000))])

# Print largest coefficients in model
pipeline_lasso.fit(train_x, train_y)
coef = pd.Series(pipeline_lasso.named_steps['lasso'].coef_, index=df_train_x.columns).sort_values()
print"The top ten most significant coefficients for lasso regression are:"
imp_coef = abs(coef).nlargest(10)
print coef.loc[imp_coef.index]

# Search for best parameters
parameters_lasso = {'lasso__alpha': [0.00008, 0.0001, 0.00012]}
optimize_and_evaluate_model(pipeline_lasso, parameters_lasso, "Lasso Regression", train_x, train_y)

########## XGBoosting Regression #############

print "XGBoosting Regression"

pipeline_xg = Pipeline(steps=[('xg',XGBRegressor(n_estimators=500, learning_rate=0.1,
                                                 max_depth=3, subsample=0.6, colsample_bytree=0.6))])

# fit the model, find the index of the top 10 features and their scores.
pipeline_xg.fit(train_x,train_y)
score = pipeline_xg.named_steps['xg'].booster().get_fscore()
sorted_score_top10 = sorted(score.items(), key=operator.itemgetter(1), reverse=True)[0:10]
index = [int(pair[0][1:]) for pair in sorted_score_top10]
feature_score = [int(pair[1]) for pair in sorted_score_top10]

# Print features by importance in model
print"The top ten most significant coefficients for random xg boost regression are:"
for i in range(len(index)):
    print df_train_x.columns[i], feature_score[i]

# Search for best parameters
parameters_xg = {'xg__subsample': (0.4, 0.6, 0.8), 'xg__colsample_bytree': (0.4, 0.6, 0.8)}
# parameters_xg = {'xg__max_depth': (3,6), 'xg__n_estimators': (150,500)}
optimize_and_evaluate_model(pipeline_xg, parameters_xg, "XGBoost Regression", train_x, train_y)


######### Random Forest Regression #############

print "Random Forest Feature Importance Analysis"

pipeline_forest = Pipeline(steps=[('randomforest',RandomForestRegressor(n_estimators=80, max_features=40, random_state=0))])

# Print features by importance in model
pipeline_forest.fit(train_x,train_y)
importances = pd.Series(pipeline_forest.named_steps['randomforest'].feature_importances_, index=df_train_x.columns).sort_values()
print"The top ten most significant coefficients for random forest regression are:"
top_importances = importances.nlargest(10)
print top_importances.cumsum()

# Search for best parameters
parameters_forest = {'randomforest__n_estimators': (60, 80),
                     'randomforest__max_features': (40, 60)}
optimize_and_evaluate_model(pipeline_forest, parameters_forest, "Random Forest Regression", train_x, train_y)

######### AdaBoosting Regression #############

print "Ada Boosting Regression"

pipeline_ada = Pipeline(steps=[('ada',AdaBoostRegressor(n_estimators=50,learning_rate=1.0))])

# Print features by importance in model
pipeline_ada.fit(train_x,train_y)
importances = pd.Series(pipeline_ada.named_steps['ada'].feature_importances_, index=df_train_x.columns).sort_values()
print"The top ten most significant coefficients for random ada boost regression are:"
top_importances = importances.nlargest(10)
print top_importances.cumsum()

# Search for best parameters
parameters_ada = {'ada__n_estimators': (40, 50, 60),
                  'ada__learning_rate': (0.1, 0.5, 1)}
optimize_and_evaluate_model(pipeline_ada, parameters_ada, "AdaBoost Regression", train_x, train_y)

########### PCA Regression #############n

pca = PCA(n_components=80)
selection = SelectKBest(k=5)
combined_features = FeatureUnion([("pca", pca), ("kbest", selection)])

pipeline_pca = Pipeline(steps=[("features", combined_features), ("lr", LinearRegression())])

# Search for best parameters
parameters_pca = {'features__pca__n_components': (20,40,80)}
optimize_and_evaluate_model(pipeline_pca, parameters_pca, "PCA plus Linear Regression", train_x, train_y)

########## Ensemble #############

print("Ensemble model, weighted average of Lasso and XGBoost Random Forest.")
for w in [0.65, 0.75, 0.85]:
    print('Lasso weight is ' + str(w))
    kf = KFold(n_splits=5, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(train_x):
        # split into train and validation
        x, x_valid = train_x[train_index], train_x[test_index]
        y, y_valid = train_y[train_index], train_y[test_index]

        # all model predictions
        lasso_prediction = pipeline_lasso.fit(x, y).predict(x_valid)
        xg_prediction = pipeline_xg.fit(x, y).predict(x_valid)

        submodel_predictions = np.vstack((lasso_prediction, xg_prediction)).T

        # make ensemble prediction
        y_fitted = lasso_prediction*w + xg_prediction*(1-w)
        rmse = np.sqrt(mean_squared_error(y_valid, y_fitted))
        scores = np.append(scores, rmse)

    print(scores)
    print("Ensemble cross validation RMSE is: " + str(np.mean(scores)))


########### Predict #############

lasso_prediction = pipeline_lasso.fit(train_x, train_y).predict(test_x)
xg_prediction = pipeline_xg.fit(train_x, train_y).predict(test_x)
ensemble_prediction = lasso_prediction*0.75 + xg_prediction*0.25

ids = list(df_test['Id'].values)
saleprices = list(np.exp(ensemble_prediction))

# write classifications to file
print("writing data to file")
myfile = open("../data/submission.csv", 'wb')
wr = csv.writer(myfile)
# header
wr.writerow(["Id", "SalePrice"])
# data rows
for id, saleprice in zip(ids, saleprices):
    wr.writerow([id, saleprice])
myfile.close()
