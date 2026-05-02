import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
from groq import Groq

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="EcoSentinel AI", layout="wide")

# -------------------------
# THEME TOGGLE
# -------------------------
theme = st.sidebar.toggle("🌙 Dark Mode")

if theme:
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🌱 EcoSentinel AI")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "⚠️ Anomaly Analysis", "🤖 AI Advisor"]
)

# -------------------------
# GROQ
# -------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# -------------------------
# DATA
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
# MODEL
# -------------------------
features = df[["water_usage","energy","chemical","temperature"]]
model = IsolationForest(contamination=0.2)
df["anomaly"] = model.fit_predict(features)

df["status"] = df["anomaly"].apply(lambda x: "🔴 Alert" if x == -1 else "🟢 Normal")
df["score"] = 100 - (df["water_usage"] * 0.05)

# -------------------------
# EVALUATION
# -------------------------
df["true_label"] = df["water_usage"].apply(lambda x: -1 if x > 800 else 1)
accuracy = accuracy_score(df["true_label"], df["anomaly"])
report = classification_report(df["true_label"], df["anomaly"], output_dict=True)

# -------------------------
# DASHBOARD
# -------------------------
if page == "📊 Dashboard":

    st.title("📊 EcoSentinel Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"### 🌿 Total Plants\n## {len(df)}")
    col2.markdown(f"### 🚨 Anomalies\n## {len(df[df['anomaly']==-1])}")
    col3.markdown(f"### ⭐ Avg Score\n## {int(df['score'].mean())}")

    st.markdown("---")

    # FILTER
    selected = st.multiselect("Select Plants", df["plant_id"], default=df["plant_id"])
    filtered_df = df[df["plant_id"].isin(selected)]

    # CHARTS
    fig = px.bar(
        filtered_df,
        x="plant_id",
        y="water_usage",
        color="status",
        title="Water Usage"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(filtered_df, x="plant_id", y="energy", markers=True, title="Energy Trend")
    st.plotly_chart(fig2, use_container_width=True)

    # TABLE with highlight
    def highlight(row):
        return ["background-color: #ff4d4d" if row["anomaly"] == -1 else "" for _ in row]

    st.dataframe(filtered_df.style.apply(highlight, axis=1))

    # DOWNLOAD
    st.download_button(
        "📄 Download Report",
        filtered_df.to_csv(index=False),
        file_name="report.csv"
    )

    st.markdown("---")

    st.subheader("📊 Model Evaluation")
    st.success(f"Accuracy: {round(accuracy*100,2)}%")
    st.json(report)

# -------------------------
# ANOMALY PAGE
# -------------------------
elif page == "⚠️ Anomaly Analysis":

    st.title("⚠️ Anomaly Insights")

    plant = st.selectbox("Select Plant", df["plant_id"])
    row = df[df["plant_id"] == plant].iloc[0]

    if row["anomaly"] == -1:
        st.error("🚨 Anomaly Detected")

        st.metric("Water Usage", row["water_usage"])
        st.metric("Energy", row["energy"])

        st.markdown("### 🔍 Root Cause")
        st.write("""
        - Cooling inefficiency  
        - Leak in pipelines  
        - Poor recycling systems  
        """)

    else:
        st.success("✅ Plant operating normally")

# -------------------------
# AI ADVISOR
# -------------------------
elif page == "🤖 AI Advisor":

    st.title("🤖 AI Sustainability Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display chat
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask your question...")

    knowledge_base = [
        "Cooling towers consume large water.",
        "Water recycling reduces usage.",
        "Leak detection prevents losses.",
        "Energy optimization improves efficiency."
    ]

    def rag_query(q):
        context = ""
        for doc in knowledge_base:
            if any(word.lower() in doc.lower() for word in q.split()):
                context += doc + "\n"

        if context == "":
            context = "\n".join(knowledge_base[:2])

        prompt = f"""
You are an industrial sustainability expert.

Context:
{context}

Question:
{q}

Give clear recommendations.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}]
        )

        return response.choices[0].message.content

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        answer = rag_query(question)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()
