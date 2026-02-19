import streamlit as st
import pandas as pd
import mlflow.sklearn
from agents.orchestrator import planner_agent
import os
from datetime import datetime

st.set_page_config(page_title="PanikPlan DineSync", page_icon="🍴", layout="wide")

# Load your specific MLflow model artifact
RUN_ID = "32f3fa1f446b4ac68bae56648e6d40cf"
EXP_ID = "450765862006708527"
model_uri = f"mlruns/{EXP_ID}/{RUN_ID}/artifacts/randomforest_model"

@st.cache_resource
def get_model():
    return mlflow.sklearn.load_model(model_uri)

model = get_model()

def save_feedback_log(name, rating, sentiment):
    fb_path = "data/user_feedback.csv"
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame([{
        "timestamp": datetime.now(),
        "name": name,
        "predicted_rating": rating,
        "sentiment": sentiment
    }])
    df.to_csv(fb_path, mode='a', header=not os.path.exists(fb_path), index=False)

st.title("🗺️ PanikPlan: Agentic Travel Planner")

with st.sidebar:
    dest = st.text_input("Destination", placeholder="e.g., Tokyo")
    budget = st.select_slider("Budget Level", options=[1, 2, 3])
    if st.button("Generate Plan"):
        inputs = {"destination": dest, "budget_level": budget, "candidates": []}
        with st.spinner("🤖 Agent searching & scoring..."):
            result = planner_agent.invoke(inputs)
            
            scored_picks = []
            for place in result["candidates"]:
                features = pd.DataFrame([{
                    "price_score": place["price_score"],
                    "number_of_reviews": place["number_of_reviews"],
                    "is_european": 0, "is_italian": 0, "is_french": 0,
                    "is_mediterranean": 0, "is_vegetarian_friendly": 0, "is_unknown": 1
                }])
                rating = model.predict(features)[0]
                place["predicted_rating"] = round(float(rating), 2)
                scored_picks.append(place)
            
            scored_picks.sort(key=lambda x: x['predicted_rating'], reverse=True)
            st.session_state.results = scored_picks

if "results" in st.session_state:
    st.subheader(f"Top Recommended Spots")
    for i, spot in enumerate(st.session_state.results):
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.write(f"🏠 **[{spot['name']}]({spot['url']})**")
        col2.metric("ML Score", f"{spot['predicted_rating']}⭐")
        
        fb = col3.feedback("thumbs", key=f"fb_{i}")
        if fb is not None:
            save_feedback_log(spot['name'], spot['predicted_rating'], fb)
            st.toast(f"Feedback saved for {spot['name']}!")