import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

# -------------------------------
# Title
# -------------------------------

st.title("EcoSentinel AI")
st.subheader("Industrial Sustainability Monitoring System")

# -------------------------------
# Simulated Plant Dataset
# -------------------------------

data = {
    "plant_id": ["P1","P2","P3","P4","P5"],
    "water_usage": [500,520,510,900,505],
    "energy": [300,290,310,320,305],
    "chemical": [50,55,52,70,51],
    "temperature": [40,39,41,45,40]
}

df = pd.DataFrame(data)

# -------------------------------
# Display Plant Data
# -------------------------------

st.write("### Plant Monitoring Data")
st.dataframe(df)

# -------------------------------
# Anomaly Detection
# -------------------------------

features = df[["water_usage","energy","chemical","temperature"]]

model = IsolationForest(contamination=0.2, random_state=42)

df["anomaly"] = model.fit_predict(features)

df["status"] = df["anomaly"].apply(lambda x: "Alert ⚠️" if x == -1 else "Normal")

st.write("### Plant Status")
st.dataframe(df[["plant_id","status"]])

# -------------------------------
# Sustainability Score
# -------------------------------

df["sustainability_score"] = 100 - (df["water_usage"] * 0.05)

# -------------------------------
# Dashboard Charts
# -------------------------------

st.subheader("Plant Sustainability Analytics Dashboard")

# Water Usage Chart
st.write("### Water Usage by Plant")

fig1 = px.bar(
    df,
    x="plant_id",
    y="water_usage",
    color="water_usage",
    title="Water Consumption Across Plants"
)

st.plotly_chart(fig1)

# Energy Consumption Chart
st.write("### Energy Consumption Trend")

fig2 = px.line(
    df,
    x="plant_id",
    y="energy",
    markers=True,
    title="Energy Consumption Across Plants"
)

st.plotly_chart(fig2)

# Sustainability Score Chart
st.write("### Sustainability Score")

fig3 = px.bar(
    df,
    x="plant_id",
    y="sustainability_score",
    color="sustainability_score",
    title="Plant Sustainability Score"
)

st.plotly_chart(fig3)

# -------------------------------
# AI Sustainability Advisor
# -------------------------------

st.write("## EcoSentinel AI Advisor")

question = st.text_input("Ask about plant sustainability:")

if not question:
    st.info("Example questions: water usage, energy optimization, chemical safety")
if question:

    if "water" in question.lower():
        st.success("""
High water usage detected.

Recommended actions:
• Inspect cooling tower efficiency  
• Implement water recycling systems  
• Check for pipeline leaks
""")

    elif "energy" in question.lower():
        st.success("""
Energy optimization recommended.

Suggested improvements:
• Install variable speed pumps  
• Optimize industrial process automation  
• Monitor energy consumption trends
""")

    elif "chemical" in question.lower():
        st.success("""
Chemical imbalance detected.

Recommended actions:
• Check chemical dosing systems  
• Calibrate monitoring sensors  
• Monitor chemical concentration levels
""")

    else:
        st.info("""
EcoSentinel AI suggests monitoring plant resources regularly
and implementing sustainability best practices.
""")
