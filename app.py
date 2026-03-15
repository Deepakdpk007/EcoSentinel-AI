import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="EcoSentinel AI", layout="wide")

st.title("EcoSentinel AI")
st.subheader("Industrial Sustainability Monitoring Platform")

# -------------------------
# Dataset
# -------------------------

data = {
    "plant_id": ["P1","P2","P3","P4","P5"],
    "water_usage": [500,520,510,900,505],
    "energy": [300,290,310,320,305],
    "chemical": [50,55,52,70,51],
    "temperature": [40,39,41,45,40]
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

col1, col2, col3 = st.columns(3)

col1.metric("Total Plants", len(df))
col2.metric("Anomalies Detected", len(df[df["status"]=="Alert ⚠️"]))
col3.metric("Avg Sustainability Score", int(df["sustainability_score"].mean()))

# -------------------------
# Charts
# -------------------------

st.write("## Sustainability Analytics")

fig1 = px.bar(df, x="plant_id", y="water_usage", title="Water Usage")
st.plotly_chart(fig1)

fig2 = px.line(df, x="plant_id", y="energy", markers=True, title="Energy Consumption")
st.plotly_chart(fig2)

fig3 = px.bar(df, x="plant_id", y="sustainability_score", title="Sustainability Score")
st.plotly_chart(fig3)

# -------------------------
# AI Explanation
# -------------------------

st.write("## AI Anomaly Explanation")

def explain(row):

    if row["status"] == "Normal":
        return "Plant operating within normal sustainability limits."

    avg = df["water_usage"].mean()

    if row["water_usage"] > avg:
        return """
High water usage detected.

Possible causes:
• cooling tower inefficiency
• pipeline leakage
• poor recycling systems
"""

    return "Abnormal resource consumption detected."

plant = st.selectbox("Select Plant", df["plant_id"])

plant_data = df[df["plant_id"]==plant].iloc[0]

st.warning(explain(plant_data))

# -------------------------
# Sustainability Advisor
# -------------------------

st.write("## Sustainability Advisor")

question = st.text_input("Ask about plant sustainability:")

def advisor(q):

    q = q.lower()

    if "water" in q:
        return """
Recommended actions:

• inspect cooling tower efficiency
• implement water recycling
• detect pipeline leaks
"""

    if "energy" in q:
        return """
Energy optimization suggestions:

• use variable speed pumps
• automate industrial processes
• monitor energy usage
"""

    if "chemical" in q:
        return """
Chemical safety improvements:

• check dosing system
• calibrate sensors
• monitor chemical concentration
"""

    return """
General sustainability improvements:

• monitor resource consumption
• improve maintenance schedules
• implement industrial monitoring systems
"""

if question:
    st.success(advisor(question))
else:
    st.info("Example questions: water usage, energy efficiency, chemical safety")
