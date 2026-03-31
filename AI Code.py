import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and target
X = data[['study_hours', 'sleep_hours', 'attendance', 'previous_marks']]
y = data['final_score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = r2_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Visualization
plt.scatter(data['study_hours'], data['final_score'])
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")
plt.show()

# User input prediction
study = float(input("Enter Study Hours: "))
sleep = float(input("Enter Sleep Hours: "))
attendance = float(input("Enter Attendance: "))
previous = float(input("Enter Previous Marks: "))

result = model.predict([[study, sleep, attendance, previous]])

print("Predicted Final Score:", result[0])