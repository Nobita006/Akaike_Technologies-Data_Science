import streamlit as st
import requests
import os

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
                
                # Display audio player if the generated file exists locally
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
