"""Main module"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

true_data = pd.read_csv("/workspace/titanic/data/gender_submission.csv")
training_data = pd.read_csv("/workspace/titanic/data/train.csv")
test_data = pd.read_csv("/workspace/titanic/data/test.csv")

features = ["Pclass", "Sex", "SibSp", "Parch"]

y = training_data["Survived"]
X = pd.get_dummies(training_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
accuracy = accuracy_score(true_data["Survived"], predictions)

if __name__ == "__main__":
    print(accuracy)
    pd.DataFrame(predictions).to_csv('results.csv', sep='\t', index=False)
