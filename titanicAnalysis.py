import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


ship_data_path = './titanic/train.csv'
ship_data = pd.read_csv(ship_data_path)
#We are trying to predict who survives.
Y = ship_data.Survived
#We utilize these columns because they contain only numerical data and they make a different for the model. The only value excluded is cabin number and names but those are unique to each passenger and our current models wouldn't be changed if they were included. Age has a lot of null values which has an impact on the predictions.
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = ship_data[features]
print(X.head())

#This line splits the code into a training set which we will use to 'teach' our model how to make predictions.
train_x, val_x, train_y, val_y = train_test_split(X, Y, random_state=0)
missing = [col for col in train_x.columns if train_x[col].isnull().any()]

#There are the Decision Tree model and the Random Forest model. The neat thing about these models is that they work with pipelines which simplifies the preprocessing steps, allows us to use cross validation, and generally shortens the code.
#model = DecisionTreeRegressor(random_state=1)
#model = RandomForestRegressor(random_state=1)
"""
pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(strategy='median')), ('model', model)])
pipeline.fit(train_x, train_y)

#Calculates the mean absolute error of the predictions. Generally we want it to be as low as possible but for some reason higher errors generated better predictions. No clue why.
prediction = pipeline.predict(val_x)
mae = mean_absolute_error(prediction, val_y)
print("Validation MAE for Model: {:,.8f}".format(mae))
#This is cross validation which finds the mean absolute error of several validation and training sets.
scores = -1 * cross_val_score(pipeline, X, Y, cv=5, scoring='neg_mean_absolute_error')
print("Cross Validation MAE for Model: {:,.8f}".format(scores.mean()))
#Generates the final prediction for models that utilize pipelines
"""
"""
test_data_path = "./titanic/test.csv"
test_data = pd.read_csv(test_data_path)
test_x = test_data[features]
test_preds = pipeline.predict(test_x)
"""
"""
#This is the gradient booster model. Since gradient boosters don't work with pipelines we have to process the data manually.
"""

#"""
my_imputer = SimpleImputer(strategy="median")
imputed_train_x = pd.DataFrame(my_imputer.fit_transform(train_x))
imputed_val_x = pd.DataFrame(my_imputer.transform(val_x))
imputed_train_x.columns = train_x.columns
imputed_val_x.columns = val_x.columns
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, early_stopping_rounds=3)
model.fit(imputed_train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
rf_prediction = model.predict(imputed_val_x)
rf_prediction = [int(predict + 0.5) for predict in rf_prediction]
rf_mae = mean_absolute_error(rf_prediction, val_y)
print("Validation MAE for Random Forest Model: {:,.8f}".format(rf_mae))

#Generates the final prediction for models that don't utilize pipelines.
test_data_path = "./titanic/test.csv"
test_data = pd.read_csv(test_data_path)
test_x = pd.DataFrame(my_imputer.transform(test_data[features]))
test_preds = model.predict(test_x)

#"""

#Generates the prediction file. The top line is to round the predictions up or down.

test_preds = [int(predict + 0.5) for predict in test_preds]
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_preds})
output.to_csv('submission.csv', index=False)
