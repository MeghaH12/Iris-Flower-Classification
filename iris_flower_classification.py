# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Step 1: Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Convert to DataFrame (optional for better visualization)
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print("First 5 rows of dataset:")
print(df.head())

# Step 3: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train the model using K-Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Step 8: Test with a custom input
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example measurements
sample_scaled = scaler.transform(sample)
predicted_class = model.predict(sample_scaled)
print(f"\nPredicted Iris Class: {iris.target_names[predicted_class][0]}")