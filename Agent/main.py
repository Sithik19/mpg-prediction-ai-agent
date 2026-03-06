from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
import pickle
import numpy as np
import os
import pandas as pd

# Initialize LLM
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
@tool
def predict_car_mpg(x1: int, x2: float, x3: float, x4: int, x5: float, x6: int, x7: int, x8: str) -> str:
    """
    Predict car MPG using SVM model with user-provided car specifications.
    
    Args:
        x1: Number of cylinders (e.g., 4, 6, 8)
        x2: Engine displacement in cubic inches (e.g., 97.0)
        x3: Horsepower (e.g., 46.0)
        x4: Vehicle weight in pounds (e.g., 1835)
        x5: Acceleration 0-60 mph in seconds (e.g., 20.5)
        x6: Model year (e.g., 70 for 1970)
        x7: Origin (1=USA, 2=Europe, 3=Japan)
        x8: Car name (e.g., 'toyota corolla')
    
    Returns:
        Predicted MPG value as a formatted string
    """
    try:
        # Load all three pickle files
        base_path = os.path.dirname(__file__) if '__file__' in globals() else '.'
        base_path = os.path.join(base_path, '..')
        
        with open(os.path.join(base_path, 'svm_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        with open(os.path.join(base_path, 'encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Encode car name
        x8_encoded = label_encoder.transform([x8])[0]
        
        # Prepare and scale input
        input_df = pd.DataFrame([[x1, x2, x3, x4, x5, x6, x7, x8_encoded]], 
                                columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8_enc'])
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        result = f"Predicted MPG: {prediction[0]:.2f} for {x8}"
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Bind tool to LLM
agent = create_agent(llm, tools=[predict_car_mpg])

res = agent.invoke({"messages": [HumanMessage(content="""Predict the output for the following features:
4, 97.0, 46.0, 1835, 20.5, 70, 1, 'toyota corolla' and List the specification """)]})    

print(res['messages'][-1].content)

