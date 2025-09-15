# app.py

import streamlit as st
import os
import sys

# Add pipeline to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the correct pipeline class we built
from atlan_helpdesk_pipeline import ChatbotPipeline

# --- Page and Pipeline Initialization ---

st.set_page_config(
    page_title="Atlan Support Copilot",
    page_icon="🤖",
    layout="wide"
)

# Use Streamlit's cache to load the pipeline only once
@st.cache_resource
def load_pipeline():
    """Load the chatbot pipeline and keep it in memory."""
    try:
        # We don't need historical data for the live app's validation context
        pipeline = ChatbotPipeline(historical_data=None)
        return pipeline
    except Exception as e:
        st.error(f"Fatal Error: Could not initialize the AI pipeline. Details: {e}")
        return None

pipeline = load_pipeline()

# --- Main App Structure ---

st.title("🤖 Atlan Support Copilot")
st.markdown("Submit a ticket to get instant analysis and AI-powered answers.")

tab1, tab2 = st.tabs(["💬 Submit & Analyze Ticket", "🔧 System Status"])

with tab1:
    if not pipeline:
        st.error("Pipeline is not available. Please check the System Status tab.")
    else:
        # Input form
        with st.form("ticket_form"):
            subject = st.text_input("Subject:", placeholder="e.g., How to connect Snowflake?")
            body = st.text_area("Detailed Description:", height=200, placeholder="Please provide as much detail as possible...")
            submitted = st.form_submit_button("🚀 Analyze Ticket", type="primary")

        if submitted and subject.strip() and body.strip():
            with st.spinner("Processing your ticket... This may take a moment."):
                # Call the pipeline's 'run' method
                result = pipeline.run(query=body, ticket_id="LIVE-TICKET")

            # Display results
            st.divider()
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("🔍 Triage Analysis")
                triage = result.triage
                st.metric("Primary Topic", triage.topics[0] if triage.topics else "N/A")
                st.metric("Priority", triage.priority)
                st.metric("Sentiment", triage.sentiment_label)
                st.metric("Confidence", f"{triage.confidence:.1%}")
                
                with st.expander("Show All Topics and Reasoning"):
                    st.write(f"**All Detected Topics:** {', '.join(triage.topics)}")
                    st.write(f"**Reasoning:** {triage.reasoning}")

            with col2:
                st.subheader("📝 AI Generated Answer")
                st.markdown(result.final_answer)

        elif submitted:
            st.warning("Please provide both a subject and a description.")

with tab2:
    st.header("🔧 System Status")
    
    st.subheader(" Environment Configuration")
    if os.getenv('GEMINI_API_KEY'):
        st.success(" GEMINI_API_KEY is configured.")
    else:
        st.error("GEMINI_API_KEY is missing. Please set it in your Streamlit secrets.")

    st.subheader(" Knowledge Base Status")
    kb_file = 'atlan_knowledge_base.json'
    if os.path.exists(kb_file):
        st.success(f" Knowledge base file ('{kb_file}') is present.")
    else:
        st.error(f" Knowledge base file ('{kb_file}') is missing from the repository.")

    st.subheader("Embeddings Cache Status")
    cache_dir = 'embeddings_cache'
    if os.path.isdir(cache_dir) and os.path.exists(f'{cache_dir}/embeddings.npy'):
        st.success(f"Embeddings cache ('{cache_dir}') is present.")
    else:
        st.error(f"Embeddings cache ('{cache_dir}') is missing. The app will not start.")