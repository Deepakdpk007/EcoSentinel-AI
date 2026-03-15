import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.title("EcoSentinel AI")
st.subheader("Industrial Sustainability Monitoring System")

# Create plant dataset
data = {
    "plant_id": ["P1","P2","P3","P4","P5"],
    "water_usage":[500,520,510,900,505],
    "energy":[300,290,310,320,305],
    "chemical":[50,55,52,70,51],
    "temperature":[40,39,41,45,40]
}

df = pd.DataFrame(data)

st.write("### Plant Monitoring Data")
st.dataframe(df)

# Anomaly detection
features = df[["water_usage","energy","chemical","temperature"]]

model = IsolationForest(contamination=0.2)

df["anomaly"] = model.fit_predict(features)

df["status"] = df["anomaly"].apply(lambda x: "Alert ⚠️" if x == -1 else "Normal")

st.write("### Plant Status")
st.dataframe(df[["plant_id","status"]])

# Recommendation
if "Alert ⚠️" in df["status"].values:
    st.error("Abnormal resource usage detected!")

    st.write("### AI Recommendation")

    st.info("""
Possible sustainability issue detected.

Recommended actions:

• Inspect cooling tower efficiency  
• Implement water recycling system  
• Check for pipeline leakage
""")
