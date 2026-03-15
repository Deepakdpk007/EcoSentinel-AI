import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
from openai import OpenAI

# -------------------------------
# Page Title
# -------------------------------

st.title("EcoSentinel AI")
st.subheader("Industrial Sustainability Monitoring System")

# -------------------------------
# Simulated Industrial Dataset
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
# Display Dataset
# -------------------------------

st.write("### Plant Monitoring Data")
st.dataframe(df)

# -------------------------------
# Machine Learning Anomaly Detection
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

# Energy Chart
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
# Knowledge Base (RAG Context)
# -------------------------------

documents = [
"""
Water recycling systems can significantly reduce industrial water usage.
Cooling towers are often responsible for high water consumption.
Leak detection systems help prevent water loss in pipelines.
""",

"""
Energy efficiency in industrial plants can be improved using variable speed pumps,
heat recovery systems, and automation of industrial processes.
""",

"""
Chemical safety requires careful monitoring of dosing levels.
Sensor calibration is important to prevent overdosing chemicals.
"""
]

# -------------------------------
# OpenAI Client
# -------------------------------

client = OpenAI()

# -------------------------------
# AI Recommendation Function
# -------------------------------

def ask_ai(question):

    context = "\n".join(documents)

    prompt = f"""
You are an industrial sustainability AI assistant.

Use the following knowledge to answer the user's question.

Knowledge Base:
{context}

User Question:
{question}

Provide clear and practical sustainability recommendations.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output[0].content[0].text

# -------------------------------
# AI Advisor Interface
# -------------------------------

st.write("## EcoSentinel AI Advisor")

question = st.text_input("Ask about plant sustainability:")

if question:
    answer = ask_ai(question)
    st.success(answer)
else:
    st.info("Example questions: How to reduce water usage? How to improve energy efficiency?")
