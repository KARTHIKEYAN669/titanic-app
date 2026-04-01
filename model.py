import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

params={
    "n_estimators":[100,200],
    "max_depth":[5,10,None]
}

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = df[features].copy()
y = df["Survived"]

X["Sex"] = X["Sex"].map({"male": 1, "female": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid=GridSearchCV(RandomForestClassifier(),params,cv=3)
grid.fit(X_train,y_train)
model=grid.best_estimator_

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", importance_df)

joblib.dump(model, "titanic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model saved successfully!")
