import streamlit as st
from bot_engine import process_query

st.set_page_config(page_title="Darukaa.Earth AI Scientist", page_icon="🌱", layout="wide")

st.title("🌱 Darukaa.Earth Biodiversity Intelligence Engine")
st.caption("AI Environmental Scientist with Grounded Multi-Metric Ecological Reasoning")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Enter your land metrics (e.g., Soil Carbon: 0.3%, Rainfall: Low, Crop: Monoculture Wheat)...")

if user_input:
    # Append and show user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
    
    with st.spinner("Analyzing ecological indicators and retrieving research..."):
        response = process_query(user_input, history)

    with st.chat_message("assistant"):
        if response.needs_clarification:
            st.warning(f"**Missing Parameters Needed:** {response.clarification_message}")
            st.session_state.messages.append({"role": "assistant", "content": response.clarification_message})
        else:
            assistant_summary = ""
            for rec in response.recommendations:
                st.subheader(f"🌾 Intervention: {rec.intervention}")
                st.markdown(f"**Scientific Mechanism:** {rec.scientific_mechanism}")
                st.markdown(f"**Time Horizon:** `{rec.time_horizon}` | **Confidence Level:** `{rec.confidence_level}`")
                
                st.markdown("**Quantified Impact on Environmental Metrics:**")
                for m in rec.impacted_metrics:
                    st.markdown(f"- **{m.metric_name}**: {m.quantified_impact}")
                
                st.caption(f"📚 *Evidence & Citations: {', '.join(rec.sources_cited)}*")
                st.divider()

                assistant_summary += f"{rec.intervention}; "

            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Provided interventions: {assistant_summary}"
            })