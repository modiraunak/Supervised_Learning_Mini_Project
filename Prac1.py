#Performing EDA and Preprocessing
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
#Load Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
#Inspect data
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
#Visualize data
sns.pairplot(df,vars=['MedInc','HouseAge','AveRooms','AveBedrms','Population','MedHouseVal'])
plt.show()
#Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
#Feature and target separation
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#Prediction
y_pred = model.predict(X_test)
#Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
#Feature importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
#Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
#Save the model
joblib.dump(model, 'random_forest_california_housing_model.pkl')
