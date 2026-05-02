import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
from groq import Groq

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="EcoSentinel AI", layout="wide")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("EcoSentinel AI")
st.sidebar.markdown("Industrial Sustainability Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Anomaly Analysis", "AI Advisor"]
)

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
# ML Model
# -------------------------
features = df[["water_usage","energy","chemical","temperature"]]
model = IsolationForest(contamination=0.2)
df["anomaly"] = model.fit_predict(features)
df["status"] = df["anomaly"].apply(lambda x: "🔴 Alert" if x == -1 else "🟢 Normal")
df["score"] = 100 - (df["water_usage"] * 0.05)

# -------------------------
# Evaluation
# -------------------------
df["true_label"] = df["water_usage"].apply(lambda x: -1 if x > 800 else 1)
accuracy = accuracy_score(df["true_label"], df["anomaly"])
report = classification_report(df["true_label"], df["anomaly"], output_dict=True)

# -------------------------
# PAGE 1 — DASHBOARD
# -------------------------
if page == "Dashboard":

    st.title("📊 EcoSentinel AI Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Plants", len(df))
    col2.metric("Anomalies", len(df[df["anomaly"] == -1]))
    col3.metric("Avg Score", int(df["score"].mean()))

    st.markdown("---")

    st.subheader("📈 Resource Usage")

    fig = px.bar(
        df,
        x="plant_id",
        y="water_usage",
        color="status",
        title="Water Usage by Plant"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(df, x="plant_id", y="energy", markers=True, title="Energy Trend")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📋 Plant Data")
    st.dataframe(df)

    st.markdown("---")
    st.subheader("📊 Model Evaluation")

    st.success(f"Accuracy: {round(accuracy*100,2)}%")
    st.json(report)

# -------------------------
# PAGE 2 — ANOMALY
# -------------------------
elif page == "Anomaly Analysis":

    st.title("⚠️ Anomaly Analysis")

    plant = st.selectbox("Select Plant", df["plant_id"])
    row = df[df["plant_id"] == plant].iloc[0]

    if row["anomaly"] == -1:
        st.error(f"{plant} is showing abnormal behavior")

        st.markdown("### Possible Causes")
        st.write("""
        - Cooling inefficiency  
        - Pipeline leakage  
        - Poor recycling system  
        """)

    else:
        st.success(f"{plant} is operating normally")

# -------------------------
# PAGE 3 — AI ADVISOR
# -------------------------
elif page == "AI Advisor":

    st.title("🤖 AI Sustainability Advisor")

    question = st.text_input("Ask your question:")

    knowledge_base = [
        "Cooling towers consume large amounts of water.",
        "Water recycling reduces usage.",
        "Leak detection prevents water loss.",
        "Energy can be optimized using smart pumps.",
        "Chemical monitoring ensures safety."
    ]

    def rag_query(q):

        context = ""
        for doc in knowledge_base:
            if any(word.lower() in doc.lower() for word in q.split()):
                context += doc + "\n"

        if context == "":
            context = "\n".join(knowledge_base[:3])

        prompt = f"""
You are an industrial sustainability expert.

Context:
{context}

Question:
{q}

Give clear recommendations.
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content":prompt}]
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"

    if question:
        answer = rag_query(question)
        st.chat_message("assistant").write(answer)
