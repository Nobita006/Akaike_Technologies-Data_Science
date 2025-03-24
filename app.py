import streamlit as st
import requests
import os
import pandas as pd
import altair as alt
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

##############################
# 1. Page Config
##############################
st.set_page_config(
    page_title="News Summarization & TTS",
    page_icon="📰",
    layout="wide"
)

##############################
# 2. Hide Streamlit Menu & Footer
##############################
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

##############################
# 3. Title & Instructions
##############################
st.title("News Summarization and Text-to-Speech Application (BBC Edition)")

st.markdown("""
### How to Use
1. Enter a company name in the sidebar (e.g., **Tesla**, **Meta**, **Apple**).
2. Click **Analyze** to fetch the latest BBC articles and run sentiment analysis.
3. View the results (final sentiment, chart, extended analysis, articles), plus an audio summary in Hindi.
4. **After** analyzing, a **Search Query (Semantic)** section appears in the sidebar. Enter a query to find relevant articles.
5. If you want to return to the original analysis view, click **Back to Full Analysis**.
""")

##############################
# 4. Sidebar Input
##############################
st.sidebar.title("Configuration")
company = st.sidebar.text_input("Enter Company Name:", "")
analyze_button = st.sidebar.button("Analyze")

# We'll store all relevant state in st.session_state
if "analysis_data" not in st.session_state:
    st.session_state["analysis_data"] = None
if "articles" not in st.session_state:
    st.session_state["articles"] = []
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = []
if "audio_file" not in st.session_state:
    st.session_state["audio_file"] = None
if "show_search_results" not in st.session_state:
    st.session_state["show_search_results"] = False
if "search_results" not in st.session_state:
    st.session_state["search_results"] = []
if "semantic_query" not in st.session_state:
    st.session_state["semantic_query"] = ""

##############################
# Local semantic model
##############################
@st.cache_resource
def load_local_model():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('all-MiniLM-L6-v2', device=device_str)

local_model = load_local_model()

def do_semantic_search(query_text, articles, embeddings, top_k=3):
    if not embeddings or not local_model:
        return []
    query_emb = local_model.encode(query_text, convert_to_tensor=True)
    emb_array = np.array(embeddings)
    scores = np.dot(emb_array, query_emb.cpu().numpy())
    top_indices = np.argsort(-scores)[:top_k]
    return [articles[i] for i in top_indices]

def display_audio(audio_file):
    # Define your backend audio base URL
    backend_audio_base_url = "http://localhost:5000/audio/"
    
    if audio_file:
        # Check if we've already fetched the audio bytes
        if "audio_bytes" not in st.session_state or st.session_state["audio_bytes"] is None:
            audio_url = f"{backend_audio_base_url}{audio_file}"
            try:
                audio_response = requests.get(audio_url)
                if audio_response.status_code == 200:
                    st.session_state["audio_bytes"] = audio_response.content
                else:
                    st.write("Audio file not available (HTTP error).")
                    return
            except Exception as e:
                st.write("Error fetching audio file:", e)
                return
        # Use the stored audio bytes for display
        st.audio(st.session_state["audio_bytes"], format="audio/mp3")
    else:
        st.write("Audio file not available.")

##############################
# Layout
##############################
col_left, col_center, col_right = st.columns([1, 5, 1])
with col_left:
    st.write("")
with col_right:
    st.write("")

with col_center:
    ######################################
    # A. User clicks "Analyze" with a company
    ######################################
    if analyze_button and company:
        api_url = "http://localhost:5000/analyze"
        payload = {"company": company}
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                data = response.json()
    
                # Save entire analysis in session state
                st.session_state["analysis_data"] = data
    
                # Extract audio_file; store in session and reset cached audio bytes
                audio_file = data.get("Audio")
                st.session_state["audio_file"] = audio_file
                st.session_state["audio_bytes"] = None  # Reset audio cache
    
                # Remove "Audio" from displayed JSON
                if "Audio" in data:
                    del data["Audio"]
    
                # Reset search states
                st.session_state["show_search_results"] = False
                st.session_state["search_results"] = []
    
                st.subheader(f"Sentiment Report of {company}")
                st.markdown(f"**{data.get('Final Sentiment Analysis', '')}**")
    
                with st.expander("Show/Hide Detailed JSON"):
                    st.json(data)
    
                # Chart
                sentiment_data = data.get("Comparative Sentiment Score", {}).get("Sentiment Distribution", {})
                if sentiment_data:
                    df = pd.DataFrame(list(sentiment_data.items()), columns=["Sentiment", "Count"])
                    chart = (
                        alt.Chart(df)
                        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                        .encode(
                            x=alt.X('Sentiment', sort=None, title="Sentiment"),
                            y=alt.Y('Count', title="Number of Articles"),
                            color='Sentiment',
                            tooltip=['Sentiment', 'Count']
                        )
                        .properties(width=600, height=400)
                    )
                    st.altair_chart(chart, use_container_width=True)
    
                # Extended Analysis
                ext_analysis = data.get("Extended Analysis", {})
                if ext_analysis:
                    st.subheader("Extended Analysis")
                    entity_counts = ext_analysis.get("Entity Counts", [])
                    word_counts = ext_analysis.get("Word Counts", [])
                    st.markdown("**Top Entities**")
                    for ent, freq in entity_counts:
                        st.write(f"{ent}: {freq}")
                    st.markdown("**Top Words**")
                    for w, freq in word_counts:
                        st.write(f"{w}: {freq}")
    
                # Articles & Embeddings
                articles = data.get("Articles", [])
                st.session_state["articles"] = articles
                st.session_state["embeddings"] = data.get("Embeddings", [])
    
                # Show articles
                if articles:
                    st.subheader("Articles")
                    for i, article in enumerate(articles, start=1):
                        with st.expander(f"Article {i}: {article.get('Title', 'No Title')}"):
                            st.write(f"**Summary:** {article.get('Summary', '')}")
                            st.write(f"**Sentiment:** {article.get('Sentiment', '')}")
                            st.write(f"**Topics:** {article.get('Topics', [])}")
    
                # Show audio TTS
                display_audio(audio_file)
    
            else:
                st.error(f"Error from API: {response.text}")
    
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")
    

    elif analyze_button and not company:
        st.error("Please enter a company name.")

    ######################################
    # B. If we already have data in session
    ######################################
    elif st.session_state["analysis_data"]:
        data = st.session_state["analysis_data"]
        articles = st.session_state["articles"]
        audio_file = st.session_state["audio_file"]

        # If we're showing search results
        if st.session_state["show_search_results"]:
            st.subheader("Semantic Search Results (Top Matches)")
            results = st.session_state["search_results"]
            if results:
                for i, art in enumerate(results, start=1):
                    with st.expander(f"Match {i}: {art['Title']}"):
                        st.write(f"**Summary:** {art['Summary']}")
                        st.write(f"**Sentiment:** {art['Sentiment']}")
                        st.write(f"**Topics:** {art['Topics']}")
            else:
                st.write("No matching articles found.")

            # Button to go back to the full analysis
            if st.button("Back to Full Analysis"):
                st.session_state["show_search_results"] = False
                st.rerun()

        else:
            # Show the full analysis
            company_name = data.get("Company", "")
            st.subheader(f"Sentiment Report of {company_name}")
            st.markdown(f"**{data.get('Final Sentiment Analysis', '')}**")

            with st.expander("Show/Hide Detailed JSON"):
                st.json(data)

            # Chart
            sentiment_data = data.get("Comparative Sentiment Score", {}).get("Sentiment Distribution", {})
            if sentiment_data:
                df = pd.DataFrame(list(sentiment_data.items()), columns=["Sentiment", "Count"])
                chart = (
                    alt.Chart(df)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                    .encode(
                        x=alt.X('Sentiment', sort=None, title="Sentiment"),
                        y=alt.Y('Count', title="Number of Articles"),
                        color='Sentiment',
                        tooltip=['Sentiment', 'Count']
                    )
                    .properties(width=600, height=400)
                )
                st.altair_chart(chart, use_container_width=True)

            # Extended Analysis
            ext_analysis = data.get("Extended Analysis", {})
            if ext_analysis:
                st.subheader("Extended Analysis")
                entity_counts = ext_analysis.get("Entity Counts", [])
                word_counts = ext_analysis.get("Word Counts", [])
                st.markdown("**Top Entities**")
                for ent, freq in entity_counts:
                    st.write(f"{ent}: {freq}")
                st.markdown("**Top Words**")
                for w, freq in word_counts:
                    st.write(f"{w}: {freq}")

            # Show articles
            if articles:
                st.subheader("Articles")
                for i, article in enumerate(articles, start=1):
                    with st.expander(f"Article {i}: {article.get('Title', 'No Title')}"):
                        st.write(f"**Summary:** {article.get('Summary', '')}")
                        st.write(f"**Sentiment:** {article.get('Sentiment', '')}")
                        st.write(f"**Topics:** {article.get('Topics', [])}")

            # Show audio TTS again
            display_audio(audio_file)

##############################
# 6. Show semantic search in sidebar if articles exist
##############################
if st.session_state["articles"]:
    st.sidebar.subheader("Search Query (Semantic)")
    st.session_state["semantic_query"] = st.sidebar.text_input("Enter Query:", key="semantic_query_box")
    search_button = st.sidebar.button("Search Articles", key="semantic_search_btn")

    if search_button and st.session_state["semantic_query"]:
        query_text = st.session_state["semantic_query"]
        articles = st.session_state["articles"]
        embeddings = st.session_state["embeddings"]
        if articles and embeddings:
            results = do_semantic_search(query_text, articles, embeddings, top_k=3)
            st.session_state["search_results"] = results
            st.session_state["show_search_results"] = True
            st.rerun()
        else:
            st.sidebar.write("No articles or embeddings available. Please Analyze first.")
