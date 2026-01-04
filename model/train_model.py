import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Load data
data = pd.read_csv("dataset/student_data.csv")

X = data[['study_hours', 'attendance', 'previous_score', 'sleep_hours']]
y = data['final_score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_error = mean_absolute_error(y_test, lr_pred)

# Model 2: Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_error = mean_absolute_error(y_test, dt_pred)

print("Linear Regression MAE:", lr_error)
print("Decision Tree MAE:", dt_error)

# Best Model Selection
if lr_error < dt_error:
    print("Best Model: Linear Regression")
else:
    print("Best Model: Decision Tree")
