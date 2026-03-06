# 🚗 MPG Prediction AI Agent

An AI-powered machine learning system that predicts **vehicle fuel efficiency (Miles Per Gallon - MPG)** using automobile specifications.  
The project integrates a **Support Vector Regression (SVR) model** with a **LangChain-based AI agent powered by Groq LLM**, enabling natural language interaction with the prediction system.

---

# 📌 Project Overview

Fuel efficiency is an important factor in automotive engineering and environmental sustainability.  
This project develops a **machine learning regression model** trained on the **Auto MPG dataset** to predict vehicle fuel efficiency based on engine and vehicle attributes.

To enhance usability, the project includes an **AI agent interface** that allows users to request predictions using **natural language queries**.

---

# 🎯 Objectives

- Build a **machine learning regression model** to predict vehicle MPG
- Implement **data preprocessing and feature engineering**
- Deploy a trained model using **serialized artifacts**
- Integrate an **AI agent with LangChain + Groq LLM**
- Enable **natural language queries for model predictions**

---

# 🧠 System Architecture

```
User Input
     │
     ▼
LangChain AI Agent
     │
     ▼
Groq LLM Processing
     │
     ▼
Machine Learning Model (SVR)
     │
     ▼
MPG Prediction Output
```

---

# 📊 Dataset

The project uses the **Auto MPG Dataset**, commonly used for regression tasks in machine learning.

### Input Features
- Cylinders
- Displacement
- Horsepower
- Weight
- Acceleration
- Model Year
- Origin

### Target Variable
**Miles Per Gallon (MPG)**

---

# 🤖 Machine Learning Models Evaluated

Multiple regression algorithms were tested:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression (SVR)

After evaluation, **Support Vector Regression (SVR)** was selected as the final model due to superior performance.

---

# ⚙️ Model Pipeline

The machine learning pipeline includes:

1. Data preprocessing
2. Feature encoding
3. Feature scaling
4. Model training
5. Model evaluation
6. Model serialization

Saved model artifacts:

```
svm_model.pkl
scaler.pkl
encoder.pkl
```

These files allow the trained model to be reused without retraining.

---

# 🧠 AI Agent Integration

The system integrates an **LLM-based AI agent** using:

- **LangChain**
- **Groq API**

The agent enables **natural language interaction with the machine learning model**.

### Example Query

```
Predict MPG for a car with:
8 cylinders
350 horsepower
weight 4000
model year 1976
```

The agent interprets the query and calls the regression model to generate predictions.

---

# 🛠 Technologies Used

| Technology | Purpose |
|-------------|---------|
| Python | Core programming language |
| Scikit-learn | Machine learning model development |
| Pandas | Data processing |
| NumPy | Numerical computation |
| LangChain | AI agent framework |
| Groq LLM | Natural language processing |
| Jupyter Notebook | Model experimentation |

---

# 📂 Project Structure

```
mpg-prediction-ai-agent
│
├── Reg_model.ipynb
├── svm_model.pkl
├── scaler.pkl
├── encoder.pkl
│
├── datasets
│   └── dataset-1.csv
│
├── Agent
│   ├── main.py
│   ├── pyproject.toml
│   └── README.md
│
└── README.md
```

---

# ▶️ Installation & Setup

## 1️⃣ Clone the repository

```bash
git clone https://github.com/Sithik19/mpg-prediction-ai-agent.git
```

## 2️⃣ Navigate to the project directory

```bash
cd mpg-prediction-ai-agent
```

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running the Project

Run the AI agent:

```bash
python main.py
```

The system will start the **LLM-powered agent interface for MPG prediction**.

---

# 📈 Future Improvements

- Build a **Streamlit web interface**
- Deploy as a **REST API using FastAPI**
- Add **model comparison dashboards**
- Improve **feature engineering and hyperparameter tuning**
- Deploy on **cloud platforms**

---

# 📚 Learning Outcomes

This project demonstrates:

- Machine learning regression modeling
- Data preprocessing pipelines
- Model serialization and reuse
- Integration of ML models with LLM agents
- Practical application of LangChain in ML systems

---

# 👨‍💻 Author

**Sithik Ranjan V R**

---

# ⭐ If you found this project useful, consider giving it a star on GitHub!
