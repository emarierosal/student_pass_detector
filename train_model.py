# train_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ['study_hours', 'attendance_percentage']
MODEL_PATH = 'student_pass_model.joblib'

def main():
    df = pd.read_csv('data/students.csv')
    X = df[FEATURES]
    y = df['passed']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(pipe, MODEL_PATH)
    print('Saved to:', MODEL_PATH)

if __name__ == '__main__':
    main()
