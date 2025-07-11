# 💡 Wafer Fault Detection – End-to-End ML System

An industrial-grade **Machine Learning** project that automates the detection of faulty semiconductor wafers using sensor data. This project simulates a **real-time QA process** and is fully deployed on the **cloud** with an interactive **web interface**.

---

## 🧾 Table of Contents

1. [📌 Problem Statement](#-problem-statement)  
2. [📊 Dataset Overview](#-dataset-overview)  
3. [🛠️ Technologies Used](#-technologies-used)  
4. [🔁 Project Workflow](#-project-workflow)  
5. [🧠 Model Training Pipeline](#-model-training-pipeline)  
6. [🌐 Web App Deployment](#-web-app-deployment)  
7. [🖥️ UI Walkthrough](#-ui-walkthrough)  
8. [📂 Project Structure](#-project-structure)  
9. [📥 Installation & Running Instructions](#-installation--running-instructions)  
10. [🚀 Live Demo](#-live-demo)  
11. [🏭 Real-World Application](#-real-world-application)  
12. [👨‍💻 Author](#-author)

---

## 📌 Problem Statement

Semiconductor wafers can often be defective due to process anomalies. Manual inspection is time-consuming and error-prone.  
The goal is to build a **machine learning system** that can classify wafers as:

- ✅ **Good Wafer**
- ❌ **Bad Wafer**

using their **sensor data** and provide **real-time prediction** via a web app.

---

## 📊 Dataset Overview

- Collected sensor data from semiconductor manufacturing equipment  
- Each row represents readings from multiple sensors  
- Final label: `Good/Bad` wafer  
- File: `wafer_data.csv`

---

## 🛠️ Technologies Used

| Category             | Tools & Frameworks                   |
|----------------------|--------------------------------------|
| Language             | Python                               |
| Libraries            | Pandas, NumPy, Scikit-learn, Matplotlib |
| Model Serialization  | Pickle                               |
| Web Framework        | Flask                                |
| Frontend             | HTML5, CSS3                          |
| IDE & Notebooks      | VS Code, Jupyter Notebook            |
| Deployment           | Render (Cloud Platform)              |

---

## 🔁 Project Workflow

1. **Data Ingestion**: Load sensor data from `.csv`
2. **EDA & Cleaning**: Explore, visualize, and preprocess data
3. **Feature Engineering**: Handle missing values, scale data
4. **Model Training**: Train classification models, evaluate performance
5. **Pipeline Integration**: Automate transformation + prediction
6. **Flask Web App**: Create web interface for interaction
7. **Cloud Deployment**: Deploy using Render

---

## 🧠 Model Training Pipeline

- Implemented using a modular structure:
  - `DataIngestion` → Reads and splits the data
  - `DataTransformation` → Preprocesses features and scales them
  - `ModelTrainer` → Trains and evaluates multiple models

- Best model selected based on **accuracy**, **R2 score**, etc.
- Final model and preprocessor saved as:
  - `artifact/model.pkl`
  - `artifact/preprocessor.pkl`

---

## 🌐 Web App Deployment

- Built a **Flask app** (`app.py`) to serve predictions
- Interface allows:
  - Uploading a `.csv` file
  - Manual sensor value input
- Final prediction shown: **Good Wafer / Bad Wafer**
- Hosted using **Render** for 24/7 availability

---

## 🖥️ UI Walkthrough

- **Homepage**: Input sensor values or upload file  
- **Output**: Displays classification result  
- **Error Handling**: Validates inputs and shows user-friendly errors

---

### 📂 Project Structure

```
wafer-fault-detection/
│
├── app.py                    # Flask web server script
├── templates/
│   └── home.html             # HTML page for user input and output
│
├── pipeline/
│   ├── train_pipeline.py     # Full model training pipeline
│   └── predict_pipeline.py   # Prediction logic using saved model
│
├── src/
│   └── components/           # Data ingestion, transformation, training modules
│
├── artifact/                 # Stores trained model, preprocessor, and raw data
│
├── notebooks/
│   ├── EDA.ipynb             # Data exploration & cleaning
│   └── model_trainer.ipynb   # Model training experimentation
│
├── wafer_data.csv            # Input dataset for training/testing
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```


2. Create Virtual Environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
    pip install -r requirements.txt

4. Run the Flask App
    python app.py


👨‍💻 Author
Arpit Kakaiya
📘 B.Tech Final Year – Computer Engineering
📍 India
📧 kakaiyaarpit@gmail.com
https://www.linkedin.com/in/arpit-kakaiya-5ab78a1a4/
