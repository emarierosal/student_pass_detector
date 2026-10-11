# predict_demo.py
import joblib

FEATURES = ['study_hours', 'attendance_percentage']
MODEL_PATH = 'student_pass_model.joblib'

def main():
    model = joblib.load(MODEL_PATH)
    cases = [
        {'study_hours': 2, 'attendance_percentage': 70},
        {'study_hours': 5, 'attendance_percentage': 70},
        {'study_hours': 9, 'attendance_percentage': 70},
    ]
    for c in cases:
        pred = model.predict([[c['study_hours'], c['attendance_percentage']]])[0]
        prob = model.predict_proba([[c['study_hours'], c['attendance_percentage']]])[0][1]
        print(f"study_hours={c['study_hours']} att={c['attendance_percentage']}: pred={pred} (pass_prob={prob:.3f}) result={'Pass' if pred==1 else 'Fail'}")

if __name__ == '__main__':
    main()
"""
Why training doesn't need to be repeated: train_model.py learns the parameters
once (fitting the Pipeline with StandardScaler + LogisticRegression on the training
data), then joblib.dump() serializes the already-fitted model object to
student_pass_model.joblib. predict_demo.py (and app.py/streamlit_app.py) use
joblib.load() to restore that exact fitted model (parameters, scaler stats, etc.)
and only apply it to new input - no retraining occurs.
"""
