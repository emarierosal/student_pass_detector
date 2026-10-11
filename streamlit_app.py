# Exercise 3: Streamlit UI
import requests
import streamlit as st
import pandas as pd
import numpy as np

API_URL = 'http://127.0.0.1:8000/predict'
FEATURES = ['study_hours', 'attendance_percentage']

st.title('Student Pass Predictor')

student_name = st.text_input('Student name')
study_hours = st.number_input('Study hours', min_value=0.0, max_value=24.0, step=0.5)
attendance = st.slider('Attendance percentage', min_value=0, max_value=100, value=70)

if st.button('Predict'):
    if not student_name.strip():
        st.warning('Please enter a student name.')
    else:
        try:
            payload = {'study_hours': float(study_hours), 'attendance_percentage': float(attendance)}
            with requests.Session() as s:
                resp = s.post(API_URL, json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                name = student_name.strip()
                if data.get('result') == 'Pass':
                    st.success(f"{name} is predicted to PASS ({data.get('pass_probability', 0):.1%} probability)")
                else:
                    st.error(f"{name} is predicted to FAIL ({data.get('pass_probability', 0):.1%} probability)")
            else:
                try:
                    err = resp.json().get('error', resp.text)
                except Exception:
                    err = resp.text
                st.error(f'API error: {err}. Start the API first: python app.py')
        except requests.exceptions.ConnectionError:
            st.error('Could not connect to Flask API at http://127.0.0.1:8000/predict. Start the API first: python app.py')
        except Exception as e:
            st.error(f'Error calling API: {e}')

st.subheader('How this prediction was generated')
st.write('Inputs sent to the API: study_hours and attendance_percentage (in order). A Logistic Regression pipeline (StandardScaler + LogisticRegression) was trained on data/students.csv. Pass when probability is at least 50%.')

st.subheader('Pass probability vs study hours (at entered attendance)')
try:
    att = float(attendance)
    hours = np.arange(0, 13, 1)
    probs = []
    with requests.Session() as s:
        for h in hours:
            try:
                r = s.post(API_URL, json={'study_hours': float(h), 'attendance_percentage': att}, timeout=5)
                if r.status_code == 200:
                    probs.append(float(r.json().get('pass_probability', np.nan)))
                else:
                    probs.append(np.nan)
            except Exception:
                probs.append(np.nan)
    dfc = pd.DataFrame({'study_hours': hours, 'pass_probability': probs})
    dfc_clean = dfc.dropna()
    if dfc_clean.empty:
        st.info('Chart unavailable (API not reachable).')
    else:
        dfc_clean['threshold'] = 0.5
        st.line_chart(dfc_clean.set_index('study_hours'))
        st.write("Threshold at 50% shown in chart context; student's input corresponds to entered values.")
except Exception as e:
    st.info(f'Chart unavailable: {e}')

