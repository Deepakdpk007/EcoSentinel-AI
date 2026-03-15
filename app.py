import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import requests

# ---------------------------
# Hugging Face API Setup
# ---------------------------

HF_API_KEY = st.secrets["HF_API_KEY"]

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# ---------------------------
# Page Title
# ---------------------------

st.title("EcoSentinel AI")
st.subheader("Industrial Sustainability Monitoring System")

# ---------------------------
# Simulated Industrial Data
# ---------------------------

data = {
    "plant_id": ["P1","P2","P3","P4","P5"],
    "water_usage": [500,520,510,900,505],
    "energy": [300,290,310,320,305],
    "chemical": [50,55,52,70,51],
    "temperature": [40,39,41,45,40]
}

df = pd.DataFrame(data)

# ---------------------------
# Show Data
# ---------------------------

st.write("### Plant Monitoring Data")
st.dataframe(df)

# ---------------------------
# ML Anomaly Detection
# ---------------------------

features = df[["water_usage","energy","chemical","temperature"]]

model = IsolationForest(contamination=0.2)

df["anomaly"] = model.fit_predict(features)

df["status"] = df["anomaly"].apply(lambda x: "Alert ⚠️" if x==-1 else "Normal")

st.write("### Plant Status")
st.dataframe(df[["plant_id","status"]])

# ---------------------------
# Sustainability Score
# ---------------------------

df["sustainability_score"] = 100 - (df["water_usage"] * 0.05)

# ---------------------------
# Charts
# ---------------------------

st.subheader("Plant Sustainability Analytics Dashboard")

# Water Usage Chart
fig1 = px.bar(df,x="plant_id",y="water_usage",title="Water Usage")
st.plotly_chart(fig1)

# Energy Chart
fig2 = px.line(df,x="plant_id",y="energy",markers=True,title="Energy Consumption")
st.plotly_chart(fig2)

# Sustainability Score Chart
fig3 = px.bar(df,x="plant_id",y="sustainability_score",title="Sustainability Score")
st.plotly_chart(fig3)

# ---------------------------
# Knowledge Base
# ---------------------------

documents = """
Water recycling systems reduce industrial water usage.
Cooling towers often cause high water consumption.
Leak detection helps prevent water loss.

Energy efficiency improves using variable speed pumps
and industrial automation systems.

Chemical safety requires proper dosing and sensor calibration.
"""

# ---------------------------
# AI Advisor Function
# ---------------------------

def ask_ai(question):

    prompt = f"""
You are an industrial sustainability expert.

Knowledge:
{documents}

Question:
{question}

Provide clear recommendations.
"""

    payload = {
        "inputs": prompt
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    result = response.json()

    return result[0]["generated_text"]

# ---------------------------
# AI Advisor UI
# ---------------------------

st.write("## EcoSentinel AI Advisor")

question = st.text_input("Ask about plant sustainability:")

if question:
    answer = ask_ai(question)
    st.success(answer)
else:
    st.info("Example: How to reduce water usage?")
