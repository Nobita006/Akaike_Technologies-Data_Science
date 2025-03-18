import streamlit as st
import requests
import os
import pandas as pd
import altair as alt

# 1. Must be the very first Streamlit command
st.set_page_config(
    page_title="News Summarization & TTS",
    page_icon="📰",
    layout="wide"  # Options: "centered" or "wide"
)

# 2. Then you can optionally hide the hamburger menu & Streamlit footer
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 3. Title & Instructions
st.title("News Summarization and Text-to-Speech Application (BBC Edition)")

st.markdown("""
### How to Use
1. Enter the name of a company (e.g., **Tesla**, **Meta**, **Apple**).
2. Click **Analyze** to fetch the latest BBC articles and run sentiment analysis.
3. View the results below, including the final sentiment analysis and an audio summary in Hindi.
""")

# 4. Sidebar for Input
st.sidebar.title("Configuration")
company = st.sidebar.text_input("Enter Company Name:", "")
analyze_button = st.sidebar.button("Analyze")

# 5. Layout with Columns (Optional)
col_left, col_center , col_right= st.columns([1, 5, 1])

with col_left:
    st.write("")
with col_right:
    st.write("")
with col_center:
    if analyze_button and company:
        api_url = "http://localhost:5000/analyze"
        payload = {"company": company}
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                # Hide the audio filename from the displayed JSON
                audio_file = data.get("Audio")
                if "Audio" in data:
                    del data["Audio"]

                st.subheader(f"Sentiment Report of {company}")
                final_sent = data.get("Final Sentiment Analysis", "")
                if final_sent:
                    st.markdown(f"**{final_sent}**")

                with st.expander("Show/Hide Detailed JSON"):
                    st.json(data)

                # Sentiment Distribution Chart
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

                # Articles in expanders
                articles = data.get("Articles", [])
                if articles:
                    st.subheader("Articles")
                    for i, article in enumerate(articles, start=1):
                        with st.expander(f"Article {i}: {article.get('Title', 'No Title')}"):
                            st.write(f"**Summary:** {article.get('Summary', '')}")
                            st.write(f"**Sentiment:** {article.get('Sentiment', '')}")
                            st.write(f"**Topics:** {article.get('Topics', [])}")

                # Audio TTS
                if audio_file and os.path.exists(audio_file):
                    st.audio(open(audio_file, 'rb').read(), format="audio/mp3")
                else:
                    st.write("Audio file not available.")
            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")
    elif analyze_button and not company:
        st.error("Please enter a company name.")
