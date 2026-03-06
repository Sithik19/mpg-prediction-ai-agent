# Auto MPG Regression Analysis – LLM Agent

## 📌 Overview
This project implements an **end-to-end regression analysis and deployment workflow** using the **Auto MPG dataset**.  
Multiple regression models are trained and evaluated, the **best-performing Support Vector Regression (SVR)** model is selected using **R² score**, serialized, and then **served through an LLM-powered agent** built with **LangChain** and **Groq**.

The system allows users to interact in **natural language**, while the agent internally invokes a trained ML regression model to generate accurate predictions.

---

## 🎯 Problem Statement
Predict the **fuel efficiency (Miles Per Gallon – MPG)** of a car based on its technical specifications such as engine size, horsepower, weight, and origin.

This is a **supervised regression problem**, as the target variable (MPG) is continuous.

---

## 📊 Dataset: Auto MPG

### Description
The Auto MPG dataset contains automobile specifications along with their fuel efficiency values.  
It is widely used for regression analysis and benchmarking machine learning models.

### Features

| Feature | Description |
|------|------------|
| `cylinders` | Number of engine cylinders |
| `displacement` | Engine displacement (cubic inches) |
| `horsepower` | Engine horsepower |
| `weight` | Vehicle weight (pounds) |
| `acceleration` | Time to accelerate from 0–60 mph |
| `model_year` | Year of manufacture |
| `origin` | Manufacturing origin (1=USA, 2=Europe, 3=Japan) |
| `car_name` | Name/model of the car |

### Target Variable
- **`mpg`** – Miles Per Gallon (continuous numerical value)

---

## 🔍 Exploratory Data Analysis (EDA)
Performed in **`Reg_model.ipynb`**, including:
- Dataset inspection and cleaning
- Feature distributions
- Relationship analysis between features and MPG
- Identification of non-linear patterns
- Preparation of features for regression modeling

EDA revealed that MPG is strongly influenced by **weight, horsepower, displacement, and cylinders**, with **non-linear relationships**.

---

## 🧠 Regression Modeling

### Models Trained
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- **Support Vector Regressor (SVR)**  

---

### Preprocessing Pipeline

#### Label Encoding
- Applied to `car_name`
- Converts categorical text into numerical values

#### Standard Scaling
- Applied to numerical features using `StandardScaler`
- Essential for distance-based models like SVR
- Ensures all features are on a comparable scale

---

### Model Evaluation
**Metric Used:** **R² Score**

- Measures how well the model explains variance in MPG
- All models were compared using the same metric
- **SVR achieved the highest R² score**

---

## 🏆 Final Model: Support Vector Regression (SVR)
SVR was chosen because:
- Captures **non-linear relationships** effectively
- Performs well with scaled features
- Generalizes better on unseen data
- Handles complex feature interactions

---

## 💾 Model Serialization

| File | Description |
|----|------------|
| `svm_model.pkl` | Trained Support Vector Regression model |
| `encoder.pkl` | LabelEncoder for `car_name` |
| `scaler.pkl` | StandardScaler for numerical features |

These artifacts ensure **consistent preprocessing and inference** during prediction.

---

## 🤖 LLM Agent Integration

### Agent Architecture
- **Framework:** LangChain  
- **LLM Provider:** Groq  
- **Model:** Qwen (via Groq API)  
- **Agent Type:** Tool-using agent  

### Tool Functionality
The LangChain tool:
- Loads the serialized SVR model, encoder, and scaler
- Encodes and scales user inputs
- Performs regression prediction
- Returns formatted prediction results

The LLM decides **when to invoke the tool** and how to present the response.

---

## ⚙️ Environment & Package Management

This project uses **`uv`**, a high-performance Python package and environment manager.

### Setup
```bash
uv init auto-mpg-regression-analysis-llm-agent
.venv\Scripts\activate   # Windows
uv add langchain langchain-groq scikit-learn pandas numpy
```
Sample input to the LLM:
```
Predict the output for the following features:
4, 97.0, 46.0, 1835, 20.5, 70, 1, 'toyota corolla' and List the specification.
```


<img width="800" height="183" alt="image" src="https://github.com/user-attachments/assets/852d48f4-5ddc-422b-b139-3f37f8c29dd5" />



Output of the LLM:


<img width="850" height="422" alt="image" src="https://github.com/user-attachments/assets/74fc873e-2134-48b7-ba9e-bcb49776a8e3" />


