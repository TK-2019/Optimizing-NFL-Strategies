from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Adjusting X and y as per your requirement
X = main_df4.drop('pass_outcome', axis=1)  # Features (dropping 'pass_outcome' column)
y = main_df4['pass_outcome']  # Target (the 'pass_outcome' column from main_df4)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Train the Logistic Regression model
log_reg.fit(x_train, y_train)

# Predict using the Logistic Regression model
y_pred = log_reg.predict(x_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)

print(f"Logistic Regression Accuracy: {accuracy:.2f}")
