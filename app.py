from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# train model on startup
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# create FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML model is running!"}

@app.get("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    flower = data.target_names[prediction[0]]
    return {"prediction": flower}
