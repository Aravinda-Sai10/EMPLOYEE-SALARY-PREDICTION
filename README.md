# EMPLOYEE-SALARY-PREDICTION

This project is Streamlit web application that predicts whether an employee earns more than 50K annually based on demographic and employment features. It uses a **Random Forest Classifier**, trained on the cleaned UCI Adult dataset.
------


## 🚀 Live Features:

-  Predicts salary class (≤50K or >50K) based on user inputs.
-  Trained using Random Forest for high accuracy.
-  Encodes categorical data using `LabelEncoder`.
-  Saves model and encoders with `joblib`.
-  Visually appealing interface with:
-  Sidebar input form
-  Clean UI

---

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier
- **Accuracy:** ~90% on test split
- **Preprocessing:**
  - Missing values removed
  - `LabelEncoder` used for categorical columns
  - Fixed column name typo: `educational-num → education-num`

----

## Screenshots
🔘 UI Interface
Replace the image below with your actual screenshot
📍 File: <img width="1914" height="895" alt="Screenshot 2025-07-23 182701" src="https://github.com/user-attachments/assets/d8a0b29b-ab94-4ad0-9c32-ab920f957b00" />

--
## 📦 Requirements:
- streamlit
- pandas
- scikit-learn
- joblib

--

## 👩‍💻 Author
- ARVA ARAVINDA SAI
- B.Tech 3rd Year – AIML
- Edunet-IBMSkills Build AIML Internship Project
- June 2025 – July 2025
