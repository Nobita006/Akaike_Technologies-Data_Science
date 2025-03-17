# News Summarization and Text-to-Speech Application

## Project Overview
This application extracts news articles related to a given company, performs sentiment analysis, conducts a comparative sentiment analysis, and generates a Hindi TTS audio summary.

## Project Structure
```
Akaike-Internship-Assignment/
├── app.py          # Streamlit frontend
├── api.py          # Flask backend API
├── utils.py        # Utility functions for news scraping, summarization, sentiment analysis, and TTS
├── requirements.txt
└── README.md       # This documentation file
```

## Features
- **News Extraction:** Scrapes at least 10 news articles using BeautifulSoup.
- **Sentiment Analysis:** Uses VADER to classify article content.
- **Comparative Analysis:** Compares sentiments across articles.
- **Text-to-Speech:** Generates Hindi audio summarizing the sentiment.
- **Web Interface:** Built with Streamlit.
- **API Communication:** Frontend and backend communicate via REST APIs.
- **Deployment:** Ready for deployment on Hugging Face Spaces.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repository_link>
   cd Akaike-Internship-Assignment
   ```

2. **Install Dependencies:**
   ```bash
   py --list
   py -3.10 -m venv env
   venv\Scripts\activate
   python --version

   pip install -r requirements.txt
   ```

3. **Run the Backend API:**
   ```bash
   python api.py
   ```
   The API will start on `http://localhost:5000`.

4. **Run the Frontend Application:**
   In another terminal, run:
   ```bash
   streamlit run app.py
   ```

5. **Usage:**
   - Enter a company name.
   - Click "Analyze" to fetch news articles, view the sentiment report, and play the Hindi TTS audio summary.

## Deployment on Hugging Face Spaces
To deploy on Hugging Face Spaces, ensure that your repository includes all required files and that the entry point (e.g., `app.py` for Streamlit or a Gradio interface) is correctly specified. Follow the [Hugging Face Spaces documentation](https://huggingface.co/spaces) for detailed deployment instructions.

## Dependencies
- Flask
- Streamlit
- Requests
- BeautifulSoup4
- NLTK
- VADER Sentiment
- gTTS

## Assumptions & Limitations
- News scraping is based on Google News; the structure may change over time.
- Topic extraction is simplified and can be enhanced using dedicated NLP techniques.
- TTS is implemented with gTTS; for more robust solutions, consider integrating advanced TTS models.

## API Documentation
- **Endpoint:** `/analyze`
  - **Method:** POST
  - **Payload:** `{"company": "Company Name"}`
  - **Response:** JSON containing the company name, articles (with title, summary, sentiment, topics), comparative sentiment analysis, final sentiment, and the audio filename.
- **Endpoint:** `/audio/<filename>`
  - **Method:** GET
  - **Purpose:** Serves the generated TTS audio file.

## Video Demo
A video demo explaining the application’s functionality is included in the repository.

---

Happy coding!
```

---

## 7. Deployment Notes

- **Local Testing:** First run the API (`python api.py`) and then the frontend (`streamlit run app.py`) locally.
- **Hugging Face Spaces:** When ready, push the repository to GitHub and link it to Hugging Face Spaces. Adjust the API URL if needed to use relative paths or a different hosting strategy.

---
