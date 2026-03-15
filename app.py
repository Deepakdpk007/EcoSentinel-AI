import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

# -----------------------
# Page Title
# -----------------------

st.title("EcoSentinel AI")
st.subheader("Industrial Sustainability Monitoring System")

# -----------------------
# Simulated Industrial Data
# -----------------------

data = {
    "plant_id": ["P1","P2","P3","P4","P5"],
    "water_usage": [500,520,510,900,505],
    "energy": [300,290,310,320,305],
    "chemical": [50,55,52,70,51],
    "temperature": [40,39,41,45,40]
}

df = pd.DataFrame(data)

st.write("### Plant Monitoring Data")
st.dataframe(df)

# -----------------------
# ML Anomaly Detection
# -----------------------

features = df[["water_usage","energy","chemical","temperature"]]

model = IsolationForest(contamination=0.2)

df["anomaly"] = model.fit_predict(features)

df["status"] = df["anomaly"].apply(lambda x: "Alert ⚠️" if x == -1 else "Normal")

st.write("### Plant Status")
st.dataframe(df[["plant_id","status"]])

# -----------------------
# Sustainability Score
# -----------------------

df["sustainability_score"] = 100 - (df["water_usage"] * 0.05)

# -----------------------
# Charts
# -----------------------

st.subheader("Plant Sustainability Analytics Dashboard")

fig1 = px.bar(df, x="plant_id", y="water_usage", title="Water Usage")
st.plotly_chart(fig1)

fig2 = px.line(df, x="plant_id", y="energy", markers=True, title="Energy Consumption")
st.plotly_chart(fig2)

fig3 = px.bar(df, x="plant_id", y="sustainability_score", title="Sustainability Score")
st.plotly_chart(fig3)

# -----------------------
# Knowledge Base
# -----------------------

knowledge = {
    "water": [
        "Inspect cooling tower efficiency",
        "Install water recycling systems",
        "Check pipelines for leaks",
        "Optimize cooling water circulation"
    ],
    "energy": [
        "Use variable speed pumps",
        "Optimize industrial process automation",
        "Install energy monitoring systems",
        "Improve equipment maintenance"
    ],
    "chemical": [
        "Check chemical dosing systems",
        "Calibrate monitoring sensors",
        "Monitor chemical concentration regularly"
    ],
    "temperature": [
        "Inspect cooling systems",
        "Check equipment overheating",
        "Improve ventilation and airflow"
    ]
}

# -----------------------
# AI Advisor Function
# -----------------------

def ask_ai(question):

    question = question.lower()

    response = []

    if "water" in question:
        response += knowledge["water"]

    if "energy" in question:
        response += knowledge["energy"]

    if "chemical" in question:
        response += knowledge["chemical"]

    if "temperature" in question:
        response += knowledge["temperature"]

    if len(response) == 0:
        response = [
            "Monitor plant resource usage regularly",
            "Improve maintenance schedules",
            "Implement sustainability monitoring systems"
        ]

    output = "Recommended actions:\n\n"

    for r in response:
        output += f"• {r}\n"

    return output

# -----------------------
# AI Advisor UI
# -----------------------

st.write("## EcoSentinel AI Advisor")

question = st.text_input("Ask about plant sustainability:")

if question:
    answer = ask_ai(question)
    st.success(answer)
else:
    st.info("Example questions: water usage, energy efficiency, chemical safety")
