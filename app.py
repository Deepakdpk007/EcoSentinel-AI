import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px

# Groq API
from groq import Groq

st.set_page_config(page_title="EcoSentinel AI", layout="wide")

st.title("EcoSentinel AI")
st.subheader("Industrial Sustainability Monitoring Platform")

# -------------------------
# Groq Setup
# -------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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

df["status"] = df["anomaly"].apply(lambda x: "Alert ⚠️" if x == -1 else "Normal")

df["sustainability_score"] = 100 - (df["water_usage"] * 0.05)

# -------------------------
# Evaluation
# -------------------------
df["true_label"] = df["water_usage"].apply(lambda x: -1 if x > 800 else 1)

accuracy = accuracy_score(df["true_label"], df["anomaly"])
report = classification_report(df["true_label"], df["anomaly"], output_dict=True)

# -------------------------
# KPI Metrics
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
# Model Evaluation
# -------------------------
st.write("## Model Evaluation")

st.write(f"Accuracy: {round(accuracy * 100, 2)}%")
st.json(report)

# -------------------------
# AI Anomaly Explanation
# -------------------------
st.write("## AI Anomaly Explanation")

plant = st.selectbox("Select Plant", df["plant_id"])
plant_data = df[df["plant_id"] == plant].iloc[0]

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

st.warning(explain(plant_data))

# -------------------------
# Simple Knowledge Base (RAG simplified)
# -------------------------
knowledge_base = [
    "Cooling towers are major consumers of water in industrial plants.",
    "Water recycling systems help reduce water waste.",
    "Leak detection systems prevent pipeline losses.",
    "Variable speed pumps improve energy efficiency.",
    "Chemical dosing systems must be calibrated regularly.",
    "Industrial monitoring improves sustainability."
]

# -------------------------
# RAG + Groq LLM
# -------------------------
def rag_query(question):

    context = ""

    for doc in knowledge_base:
        if any(word.lower() in doc.lower() for word in question.split()):
            context += doc + "\n"

    if context == "":
        context = "\n".join(knowledge_base[:3])

    prompt = f"""
You are an industrial sustainability expert.

Use this knowledge to answer the question.

Knowledge:
{context}

Question:
{question}

Provide clear sustainability recommendations.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API Error: {str(e)}"

# -------------------------
# AI Sustainability Advisor
# -------------------------
st.write("## AI Sustainability Advisor")

question = st.text_input("Ask about plant sustainability:")

if question:
    answer = rag_query(question)
    st.success(answer)
else:
    st.info("Example: How can a plant reduce water usage?")
