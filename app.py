
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
data = load_iris()
X, y = data.data, data.target

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model trained successfully!")
print(f"Accuracy: {accuracy * 100:.2f}%")
