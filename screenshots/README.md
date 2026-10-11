# Screenshot Submission Guide for Internship Demonstration

Follow these steps to capture the 5 genuine screenshots required for your internship / project demonstration report.

---

### Screenshot 1: Home Page Before Symptom Selection
- **File Name:** `screenshots/1_home_page_initial.png`
- **How to capture:**
  1. Start the Flask server: `py app.py`
  2. Open your browser to `http://127.0.0.1:5000/`
  3. Ensure the title **HealthAI Predictor**, the Medical Disclaimer banner, and the symptom categories are clearly visible.
  4. Capture full window or viewport screenshot.

---

### Screenshot 2: Home Page with Symptoms Selected & Filtered
- **File Name:** `screenshots/2_home_page_symptoms_selected.png`
- **How to capture:**
  1. On `http://127.0.0.1:5000/`, type `fever` in the search box to demonstrate the live filter.
  2. Check the boxes for **Continuous Sneezing**, **Cough**, and **Mild Fever**.
  3. Notice that the selection counter in the top-right updates to `Selected: 3 / 27`.
  4. Capture the browser window showing the checked items and active counter.

---

### Screenshot 3: Prediction Result Page with Real ML Output
- **File Name:** `screenshots/3_prediction_result_page.png`
- **How to capture:**
  1. Click **Analyze Symptoms & Predict Disease**.
  2. The browser navigates to `/predict` showing **Common Cold** inside the prediction box.
  3. Ensure the active symptom tags (**Continuous Sneezing**, **Cough**, **Mild Fever**) and the **IMPORTANT MEDICAL NOTICE** disclaimer banner are visible.
  4. Capture the screenshot.

---

### Screenshot 4: Model Training and Evaluation Terminal Output
- **File Name:** `screenshots/4_terminal_model_training.png`
- **How to capture:**
  1. In PowerShell / Command Prompt, run: `py train_model.py`
  2. Ensure the terminal shows:
     - Dataset split (230 train, 58 test)
     - Decision Tree accuracy (82.76%)
     - Naive Bayes benchmark accuracy (98.28%)
     - Top informative symptoms
     - Success confirmation message
  3. Capture the terminal window.

---

### Screenshot 5: Automated Pytest Suite Passing
- **File Name:** `screenshots/5_terminal_pytest_verification.png`
- **How to capture:**
  1. In PowerShell / Command Prompt, run: `py -m pytest tests/ -v`
  2. Ensure the terminal shows all 15 tests passing: `15 passed in 1.38s`.
  3. Capture the terminal window.
