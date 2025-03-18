import streamlit as st
import requests
import os
import pandas as pd

st.title("News Summarization and Text-to-Speech Application (BBC Edition)")

company = st.text_input("Enter Company Name:", "")

if st.button("Analyze"):
    if company:
        api_url = "http://localhost:5000/analyze"  # Adjust if needed
        payload = {"company": company}
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                st.subheader("Sentiment Report")
                st.json(data)
                
                # Visualize sentiment distribution using a bar chart
                sentiment_data = data.get("Comparative Sentiment Score", {}).get("Sentiment Distribution", {})
                if sentiment_data:
                    df = pd.DataFrame(list(sentiment_data.items()), columns=["Sentiment", "Count"]).set_index("Sentiment")
                    st.bar_chart(df)
                
                # Display the final sentiment analysis text
                st.subheader("Final Sentiment Analysis")
                st.write(data.get("Final Sentiment Analysis", ""))
                
                # Display audio player for the Hindi TTS output
                audio_file = data.get("Audio")
                if audio_file and os.path.exists(audio_file):
                    with open(audio_file, 'rb') as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.write("Audio file not available.")
            else:
                st.error("Error from API: " + response.text)
        except Exception as e:
            st.error("Error connecting to API: " + str(e))
    else:
        st.error("Please enter a company name.")
