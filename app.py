import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Sample dataset for fitness tracking
data = {
    'Steps': [3000, 5000, 7000, 10000, 12000, 15000],
    'Calories_Burned': [150, 230, 310, 450, 520, 600],
    'Workout_Duration': [15, 25, 35, 50, 60, 75],
    'Heart_Rate': [80, 85, 90, 95, 100, 110]
}

df = pd.DataFrame(data)
print("Sample Fitness Data:")
print(df.head())

# Data Visualization
sns.pairplot(df)
plt.show()

# Splitting data for machine learning
X = df[['Steps', 'Workout_Duration', 'Heart_Rate']]
y = df['Calories_Burned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Machine Learning Model - Predicting Calories Burned
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Model Performance:")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
