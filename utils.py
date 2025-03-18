import os
import logging
import time
import uuid
import asyncio
import aiohttp
import requests
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gtts import gTTS
from typing import List, Dict, Tuple

# Set HF hub warning variable for Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Import torch to check for GPU availability
import torch
device = 0 if torch.cuda.is_available() else -1  # device=0 uses GPU; -1 uses CPU

# Additional libraries for advanced summarization, topic extraction, translation, and sentiment analysis
from transformers import pipeline
import spacy
from googletrans import Translator  # Use googletrans==4.0.0-rc1 for best results
from keybert import KeyBERT

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model (download if necessary)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    logging.info("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize transformer pipelines with explicit models and device settings
try:
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        revision="a4f8f3e",
        device=device
    )
except Exception as e:
    logging.error(f"Error loading transformer summarization pipeline: {e}")
    summarizer = None

try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
except Exception as e:
    logging.error(f"Error loading transformer sentiment analyzer: {e}")
    sentiment_pipeline = None

# Initialize KeyBERT for advanced topic extraction
try:
    kw_model = KeyBERT()
except Exception as e:
    logging.error(f"Error initializing KeyBERT: {e}")
    kw_model = None

# Initialize translator for converting text to Hindi
translator = Translator()

# Simple in-memory cache with expiry (5 minutes)
cache = {}
CACHE_EXPIRY = 300  # seconds

def get_from_cache(key: str):
    if key in cache:
        entry = cache[key]
        if time.time() - entry["time"] < CACHE_EXPIRY:
            return entry["data"]
    return None

def set_cache(key: str, data):
    cache[key] = {"data": data, "time": time.time()}

async def fetch_article_content(session: aiohttp.ClientSession, url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            paragraphs = soup.find_all('p')
            if paragraphs:
                combined = " ".join(p.get_text(strip=True) for p in paragraphs[:3])
                return combined
            else:
                return ""
    except Exception as e:
        logging.error(f"Error fetching article content asynchronously from {url}: {e}")
        return ""

def fetch_news(company_name: str, num_articles: int = 10) -> List[Dict[str, str]]:
    """
    Fetch BBC news articles related to the given company.
    Requests 15 results and then slices to the desired number.
    """
    cache_key = f"bbc_{company_name}_{num_articles}"
    cached = get_from_cache(cache_key)
    if cached:
        logging.info("Returning cached news data")
        return cached

    base_url = "https://www.bbc.com/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        params = {"q": company_name}
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to fetch BBC search results for '{company_name}': {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    # Request more results than needed, then slice.
    result_cards = soup.find_all('div', attrs={"data-testid": "newport-card"}, limit=15)
    logging.info(f"Found {len(result_cards)} BBC search results for '{company_name}'.")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def process_card(card):
        link_tag = card.find('a', attrs={"data-testid": "internal-link"})
        link = None
        if link_tag and link_tag.has_attr('href'):
            href = link_tag['href']
            link = "https://www.bbc.com" + href if href.startswith('/') else href
        title_tag = card.find('h2', attrs={"data-testid": "card-headline"})
        title = title_tag.get_text(strip=True) if title_tag else "No Title Found"
        snippet_tag = card.find('div', class_='sc-4ea10043-3')
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        summary = snippet
        if link:
            async with aiohttp.ClientSession() as session:
                content = await fetch_article_content(session, link)
                if content and len(content) > len(snippet):
                    summary = content
        return {"Title": title, "Link": link, "Summary": summary if summary else "Summary not available"}
    
    tasks = [process_card(card) for card in result_cards]
    articles = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    
    articles = articles[:num_articles]
    set_cache(cache_key, articles)
    return articles

def summarize_text(text: str, num_sentences: int = 3) -> str:
    """
    Basic extractive summarization using the first few sentences.
    """
    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences]) if sentences else text

def advanced_summarize(text: str, num_sentences: int = 1) -> str:
    """
    Use transformer-based summarization if available and text is long enough;
    otherwise, fallback to basic summarization.
    """
    if summarizer and len(text.split()) > 50:
        try:
            input_length = len(text.split())
            max_len = 130 if input_length >= 130 else input_length
            summary = summarizer(text, max_length=max_len, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logging.error(f"Transformer summarization error: {e}")
    return summarize_text(text, num_sentences)

def analyze_sentiment(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Analyze sentiment using a transformer-based model if available; otherwise fallback to VADER.
    """
    if sentiment_pipeline:
        try:
            result = sentiment_pipeline(text)[0]
            label = result.get("label", "NEUTRAL").upper()
            score = result.get("score", 0.0)
            sentiment = "Positive" if label == "POSITIVE" else "Negative" if label == "NEGATIVE" else "Neutral"
            return sentiment, {"compound": score, "raw": result}
        except Exception as e:
            logging.error(f"Transformer sentiment analysis error: {e}")
    analyzer_vader = SentimentIntensityAnalyzer()
    scores = analyzer_vader.polarity_scores(text)
    compound = scores.get('compound', 0)
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, scores

def extract_topics(text: str, num_topics: int = 5) -> List[str]:
    """
    Extract key topics using KeyBERT. Falls back to spaCy noun-chunk extraction if necessary.
    """
    if kw_model:
        try:
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_topics)
            topics = [kw[0] for kw in keywords]
            return topics
        except Exception as e:
            logging.error(f"KeyBERT topic extraction error: {e}")
    doc = nlp(text)
    chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
    freq = {}
    for chunk in chunks:
        freq[chunk] = freq.get(chunk, 0) + 1
    sorted_chunks = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    topics = [chunk for chunk, _ in sorted_chunks[:num_topics]]
    return topics

def translate_to_hindi(text: str) -> str:
    """
    Translate the provided text to Hindi using googletrans.
    """
    try:
        translated = translator.translate(text, dest='hi')
        return translated.text
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

def process_article(article: Dict[str, str]) -> Dict:
    """
    Process an article: advanced summarization, sentiment analysis, and topic extraction.
    """
    original_summary = article.get("Summary", "")
    if original_summary and original_summary != "Summary not available":
        summarized_text = advanced_summarize(original_summary)
        sentiment, scores = analyze_sentiment(summarized_text)
        topics = extract_topics(summarized_text)
    else:
        summarized_text = original_summary
        sentiment, scores = "Neutral", {"compound": 0.0}
        topics = []
    return {
        "Title": article.get("Title") or "No Title Found",
        "Summary": summarized_text,
        "Sentiment": sentiment,
        "Sentiment Scores": scores,
        "Topics": topics
    }

def comparative_analysis(articles: List[Dict]) -> Dict:
    """
    Compute comparative sentiment analysis and topic overlap across all articles.
    This groups articles by sentiment and computes:
      - Sentiment Distribution.
      - Coverage Differences: Compare positive vs. negative articles and provide an overall comparison.
      - Topic Overlap: Common topics between positive and negative articles, and unique topics in each group.
    """
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for art in articles:
        sentiment = art.get("Sentiment", "Neutral")
        sentiment_counts[sentiment] += 1

    # Group article titles and topics by sentiment
    pos_titles = [art["Title"] for art in articles if art["Sentiment"] == "Positive"]
    neg_titles = [art["Title"] for art in articles if art["Sentiment"] == "Negative"]
    
    # Create comparison statements across all articles
    comparisons = []
    if pos_titles and neg_titles:
        comparisons.append({
            "Comparison": f"Positive articles ({', '.join(pos_titles)}) highlight opportunities, while negative articles ({', '.join(neg_titles)}) emphasize challenges.",
            "Impact": "This indicates mixed market sentiment."
        })
    all_titles = [art["Title"] for art in articles]
    comparisons.append({
        "Comparison": f"Overall, the coverage spans: {', '.join(all_titles)}.",
        "Impact": "The news reflects a broad spectrum of perspectives."
    })
    
    # Compute common topics between positive and negative groups
    pos_topics = set().union(*(set(art["Topics"]) for art in articles if art["Sentiment"] == "Positive"))
    neg_topics = set().union(*(set(art["Topics"]) for art in articles if art["Sentiment"] == "Negative"))
    common_topics = list(pos_topics.intersection(neg_topics))
    unique_pos = list(pos_topics - set(common_topics))
    unique_neg = list(neg_topics - set(common_topics))
    
    return {
        "Sentiment Distribution": sentiment_counts,
        "Coverage Differences": comparisons,
        "Topic Overlap": {
            "Common Topics": common_topics,
            "Unique Topics in Positive Articles": unique_pos,
            "Unique Topics in Negative Articles": unique_neg
        }
    }

def generate_tts(text: str, lang: str = 'hi') -> str:
    """
    Translate text to Hindi and generate TTS audio using gTTS.
    """
    hindi_text = translate_to_hindi(text)
    try:
        tts = gTTS(text=hindi_text, lang=lang)
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        tts.save(filename)
        logging.info(f"TTS generated and saved as {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error generating TTS: {e}")
        return ""
