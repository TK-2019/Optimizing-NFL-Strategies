import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(max_depth=32, max_features='log2', min_samples_leaf=15,
                       min_samples_split=10, n_estimators=111)  # n_estimators is the number of trees

X = main_df4.drop(columns=['pass_outcome'])
y = main_df4['pass_outcome']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Fit the model
rf_model.fit(x_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(x_test)
y_pred_train = rf_model.predict(x_train)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_train = accuracy_score(y_train, y_pred_train)

print("Random Forest Test Accuracy:", accuracy_rf)
print("Random Forest Train Accuracy:", accuracy_train)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)

# Plotting the confusion matrix with a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'],
            annot_kws={"size": 16})  # Increase the font size of the numbers inside the heatmap

# Add labels, title, and axis ticks with larger font size
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix with Heatmap', fontsize=16)

# Increase the font size of the axis ticks
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.show()

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Define the objective function for optimization
def objective(trial):
    rf = RandomForestClassifier(
        n_estimators=trial.suggest_int('n_estimators', 50, 200),
        max_depth=trial.suggest_int('max_depth', 10, 50),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None])  # Removed 'auto'
    )
    score = cross_val_score(rf, x_train, y_train, n_jobs=-1, cv=5, scoring='accuracy')
    return score.mean()

# Run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
best_rf = RandomForestClassifier(**best_params)
best_rf.fit(x_train, y_train)
