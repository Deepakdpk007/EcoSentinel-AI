import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
from transformers import pipeline

st.set_page_config(page_title="EcoSentinel AI", layout="wide")

st.title("EcoSentinel AI")
st.subheader("Industrial Sustainability Monitoring Platform")

# -------------------------
# Dataset
# -------------------------

data = {
    "plant_id":["P1","P2","P3","P4","P5"],
    "water_usage":[500,520,510,900,505],
    "energy":[300,290,310,320,305],
    "chemical":[50,55,52,70,51],
    "temperature":[40,39,41,45,40]
}

df = pd.DataFrame(data)

# -------------------------
# ML Anomaly Detection
# -------------------------

features = df[["water_usage","energy","chemical","temperature"]]

model = IsolationForest(contamination=0.2)

df["anomaly"] = model.fit_predict(features)

df["status"] = df["anomaly"].apply(lambda x: "Alert ⚠️" if x==-1 else "Normal")

df["sustainability_score"] = 100 - (df["water_usage"]*0.05)

# -------------------------
# KPI Cards
# -------------------------

col1,col2,col3 = st.columns(3)

col1.metric("Total Plants",len(df))
col2.metric("Anomalies Detected",len(df[df["status"]=="Alert ⚠️"]))
col3.metric("Avg Sustainability Score",int(df["sustainability_score"].mean()))

# -------------------------
# Charts
# -------------------------

st.write("## Sustainability Analytics")

fig1 = px.bar(df,x="plant_id",y="water_usage",title="Water Usage")
st.plotly_chart(fig1)

fig2 = px.line(df,x="plant_id",y="energy",markers=True,title="Energy Consumption")
st.plotly_chart(fig2)

fig3 = px.bar(df,x="plant_id",y="sustainability_score",title="Sustainability Score")
st.plotly_chart(fig3)

# -------------------------
# Load LLM
# -------------------------

@st.cache_resource
def load_llm():
    return pipeline("text2text-generation",model="google/flan-t5-base")

llm = load_llm()

# -------------------------
# Knowledge Context
# -------------------------

knowledge = """
Water recycling systems reduce industrial water usage.
Cooling towers often cause high water consumption.
Leak detection systems prevent pipeline losses.

Energy efficiency improves using variable speed pumps
and industrial automation systems.

Chemical safety requires proper dosing and sensor calibration.
"""

# -------------------------
# AI Advisor
# -------------------------

st.write("## AI Sustainability Advisor")

question = st.text_input("Ask about plant sustainability:")

def ask_ai(question):

    prompt = f"""
Using the following sustainability knowledge:

{knowledge}

Answer this question:
{question}
"""

    result = llm(prompt,max_length=150)

    return result[0]["generated_text"]

if question:
    answer = ask_ai(question)
    st.success(answer)

else:
    st.info("Example: How can a plant reduce water consumption?")
