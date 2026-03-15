import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import requests

st.set_page_config(page_title="EcoSentinel AI", layout="wide")

# HuggingFace Router
HF_API_KEY = st.secrets["HF_API_KEY"]

API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# --------------------
# Title
# --------------------

st.title("EcoSentinel AI")
st.subheader("Industrial Sustainability Monitoring Platform")

# --------------------
# Dataset
# --------------------

data = {
    "plant_id":["P1","P2","P3","P4","P5"],
    "water_usage":[500,520,510,900,505],
    "energy":[300,290,310,320,305],
    "chemical":[50,55,52,70,51],
    "temperature":[40,39,41,45,40]
}

df = pd.DataFrame(data)

# --------------------
# ML Anomaly Detection
# --------------------

features = df[["water_usage","energy","chemical","temperature"]]

model = IsolationForest(contamination=0.2)

df["anomaly"] = model.fit_predict(features)

df["status"] = df["anomaly"].apply(lambda x: "Alert ⚠️" if x==-1 else "Normal")

df["sustainability_score"] = 100 - (df["water_usage"]*0.05)

# --------------------
# KPI Cards
# --------------------

col1,col2,col3 = st.columns(3)

col1.metric("Total Plants",len(df))

col2.metric("Anomalies",len(df[df["status"]=="Alert ⚠️"]))

col3.metric("Avg Sustainability Score",int(df["sustainability_score"].mean()))

# --------------------
# Charts
# --------------------

st.write("## Sustainability Analytics")

fig1 = px.bar(df,x="plant_id",y="water_usage",title="Water Usage")
st.plotly_chart(fig1)

fig2 = px.line(df,x="plant_id",y="energy",markers=True,title="Energy Consumption")
st.plotly_chart(fig2)

fig3 = px.bar(df,x="plant_id",y="sustainability_score",title="Sustainability Score")
st.plotly_chart(fig3)

# --------------------
# Knowledge Context
# --------------------

knowledge = """
Water recycling systems reduce industrial water usage.
Cooling towers often cause excessive water consumption.
Leak detection systems prevent pipeline losses.

Energy efficiency can improve using variable speed pumps
and industrial automation.

Chemical safety requires correct dosing and sensor calibration.
"""

# --------------------
# LLM Advisor
# --------------------

def ask_ai(question):

    prompt = f"""
You are an industrial sustainability expert.

Knowledge:
{knowledge}

Question:
{question}

Provide clear recommendations.
"""

    payload = {"inputs":prompt}

    try:
        r = requests.post(API_URL,headers=headers,json=payload)

        if r.status_code != 200:
            return f"API Error: {r.text}"

        try:
            result = r.json()
        except:
            return "Model loading. Try again in a few seconds."

        if isinstance(result,list):
            return result[0].get("generated_text","No answer generated.")

        return str(result)

    except Exception as e:
        return str(e)

# --------------------
# AI Advisor UI
# --------------------

st.write("## AI Sustainability Advisor")

question = st.text_input("Ask about plant sustainability:")

if question:
    response = ask_ai(question)
    st.success(response)
else:
    st.info("Example: Why is water usage high?")
