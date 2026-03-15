import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import os

# --- Modern GenAI Imports (Upgraded to LCEL) ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="EcoSentinel AI", layout="wide")
st.title("🌿 EcoSentinel AI")
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
model = IsolationForest(contamination=0.2, random_state=42)
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
st.write("## 📊 Sustainability Analytics")
colA, colB = st.columns(2)

with colA:
    fig1 = px.bar(df, x="plant_id", y="water_usage", color="status", 
                  color_discrete_map={"Normal": "#28a745", "Alert ⚠️": "#dc3545"}, title="Water Usage")
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    fig3 = px.bar(df, x="plant_id", y="sustainability_score", title="Sustainability Score")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# 🧠 AI Advisor (GenAI + Modern RAG Architecture)
# -------------------------
st.divider()
st.write("## 🧠 True GenAI Sustainability Advisor")
st.info("This section uses LangChain LCEL, FAISS, and an LLM to read the plant's engineering manual and generate dynamic troubleshooting steps.")

hf_token = st.text_input("Enter your Hugging Face Token to activate AI:", type="password")
question = st.text_input("Ask the AI about plant troubleshooting (e.g., 'Plant P4 has water usage of 900. What should we do?'):")

if hf_token and question:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    
    with st.spinner("AI is reading the engineering manual and thinking..."):
        try:
            # 1. Create a simulated manual
            manual_text = """
            EcoSentinel Industrial Plant Troubleshooting Manual:
            Section 1 - Cooling Systems: If water usage exceeds 800 units, it indicates a severe cooling tower leak or a malfunctioning evaporation valve. Engineers must immediately shut off the main water line and inspect the secondary cooling loops.
            Section 2 - Energy: If energy consumption spikes above 350 units without a production increase, check the primary compressors for friction wear.
            Section 3 - Chemicals: High chemical concentration usually means the filtration membranes are degrading and need replacement.
            """
            with open("manual.txt", "w") as f:
                f.write(manual_text)

            # 2. Document Ingestion
            loader = TextLoader("manual.txt")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            chunks = text_splitter.split_documents(documents)

            # 3. Vector Database
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_db = FAISS.from_documents(chunks, embeddings)

            # 4. Connect to LLM
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.3,
                max_new_tokens=250
            )

            # 5. Modern RAG Chain Execution (LCEL)
            prompt = PromptTemplate.from_template(
                "Use the following manual to answer the question.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:"
            )
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vector_db.as_retriever(search_kwargs={"k": 2})
            qa_chain = create_retrieval_chain(retriever, document_chain)

            response = qa_chain.invoke({"input": question})
            
            st.success("🤖 AI Diagnostic Complete:")
            st.write(response['answer'])
            
        except Exception as e:
            st.error(f"An error occurred: {e}. Please check your token or reboot the app.")

elif question and not hf_token:
    st.warning("🔒 Please enter a valid Hugging Face API token to unlock the AI Advisor.")
