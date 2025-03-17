import logging
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gtts import gTTS
import uuid
from typing import List, Dict, Tuple

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download necessary NLTK data files (if not already available)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def fetch_news(company_name: str, num_articles: int = 10) -> List[Dict[str, str]]:
    """
    Fetch news articles related to the given company from BBC's search page.
    We parse the search results for each article's link, title, and snippet.
    Then we make a second request to the article page to fetch paragraphs.
    
    Args:
        company_name (str): The name of the company or search keyword.
        num_articles (int): The maximum number of articles to fetch.
    
    Returns:
        List[Dict[str, str]]: A list of dictionaries with keys:
                              "Title", "Link", "Summary".
    """
    # BBC search URL
    base_url = "https://www.bbc.com/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # Make the search request
    try:
        params = {"q": company_name}
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to fetch BBC search results for '{company_name}': {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Each search result is typically in a div[data-testid="newport-card"]
    result_cards = soup.find_all('div', attrs={"data-testid": "newport-card"}, limit=num_articles)
    logging.info(f"Found {len(result_cards)} BBC search results for '{company_name}'.")
    
    articles = []
    for card in result_cards:
        # Extract the link from <a data-testid="internal-link" ...>
        link_tag = card.find('a', attrs={"data-testid": "internal-link"})
        link = None
        if link_tag and link_tag.has_attr('href'):
            href = link_tag['href']
            # Convert relative links (e.g., /news/articles/...) to absolute
            if href.startswith('/'):
                link = "https://www.bbc.com" + href
            else:
                link = href
        
        # Extract the headline from <h2 data-testid="card-headline">
        title_tag = card.find('h2', attrs={"data-testid": "card-headline"})
        title = title_tag.get_text(strip=True) if title_tag else "No Title Found"
        
        # Extract the snippet from <div class="sc-4ea10043-3 kMizuB"> 
        # (class may change, so watch for the textual region)
        snippet_tag = card.find('div', class_='sc-4ea10043-3')
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        
        # Attempt to fetch full text from the article page
        summary = snippet
        if link:
            try:
                article_resp = requests.get(link, headers=headers, timeout=10)
                article_resp.raise_for_status()
                article_soup = BeautifulSoup(article_resp.text, 'html.parser')
                
                # BBC articles often have multiple <p> tags in main content
                paragraphs = article_soup.find_all('p')
                if paragraphs:
                    # Combine the first 3 paragraphs for a more detailed summary
                    combined = " ".join(p.get_text(strip=True) for p in paragraphs[:3])
                    # If we found something meaningful, override the snippet
                    if len(combined) > len(snippet):
                        summary = combined
            except Exception as e:
                logging.error(f"Error fetching BBC article content from {link}: {e}")
        
        articles.append({
            "Title": title,
            "Link": link,
            "Summary": summary if summary else "Summary not available"
        })
    
    return articles


def summarize_text(text: str, num_sentences: int = 3) -> str:
    """
    Generate a simple summary by returning the first few sentences.
    
    Args:
        text (str): The text to summarize.
        num_sentences (int): Number of sentences to include.
    
    Returns:
        str: A summarized version of the text.
    """
    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences]) if sentences else text


def analyze_sentiment(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Analyze the sentiment of the provided text using VADER.
    
    Args:
        text (str): The text to analyze.
    
    Returns:
        Tuple[str, Dict[str, float]]: Sentiment label and detailed scores.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
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
    Extract key topics from text by identifying the most frequent nouns.
    
    Args:
        text (str): The text to extract topics from.
        num_topics (int): Number of topics to return.
    
    Returns:
        List[str]: A list of extracted topics.
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    # Filter for nouns (common and proper)
    nouns = [word.lower() for word, tag in tagged if tag in ('NN', 'NNS', 'NNP', 'NNPS')]
    freq = {}
    for noun in nouns:
        freq[noun] = freq.get(noun, 0) + 1
    sorted_nouns = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    topics = [noun for noun, _ in sorted_nouns[:num_topics]]
    return topics


def process_article(article: Dict[str, str]) -> Dict:
    """
    Process a raw article by summarizing, performing sentiment analysis, and extracting topics.
    
    Args:
        article (Dict[str, str]): An article with "Title", "Link", and "Summary".
    
    Returns:
        Dict: Processed article with updated "Summary", "Sentiment", "Sentiment Scores", and "Topics".
    """
    original_summary = article.get("Summary", "")
    if original_summary and original_summary != "Summary not available":
        summarized_text = summarize_text(original_summary)
        sentiment, scores = analyze_sentiment(summarized_text)
        topics = extract_topics(summarized_text)
    else:
        summarized_text = original_summary
        sentiment, scores = "Neutral", {"compound": 0.0}
        topics = []
    
    processed = {
        "Title": article.get("Title") or "No Title Found",
        "Link": article.get("Link"),
        "Summary": summarized_text,
        "Sentiment": sentiment,
        "Sentiment Scores": scores,
        "Topics": topics
    }
    return processed


def comparative_analysis(articles: List[Dict]) -> Dict:
    """
    Perform comparative sentiment analysis across articles.
    
    Args:
        articles (List[Dict]): List of processed article dictionaries.
    
    Returns:
        Dict: Summary of sentiment distribution, coverage differences, and topic overlap.
    """
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for art in articles:
        sentiment = art.get("Sentiment", "Neutral")
        sentiment_counts[sentiment] += 1
    
    comparisons = []
    if len(articles) >= 2:
        comp_text = (
            f"Article 1 '{articles[0].get('Title', 'No Title')}' is {articles[0].get('Sentiment', 'Neutral')} while "
            f"Article 2 '{articles[1].get('Title', 'No Title')}' is {articles[1].get('Sentiment', 'Neutral')}."
        )
        comparisons.append({
            "Comparison": comp_text,
            "Impact": "The differences in sentiment indicate varied media perspectives."
        })
    
    # Compare topics between the first two articles (if available)
    topics_article1 = articles[0].get("Topics", []) if len(articles) > 0 else []
    topics_article2 = articles[1].get("Topics", []) if len(articles) > 1 else []
    common_topics = list(set(topics_article1) & set(topics_article2))
    unique_topics_1 = list(set(topics_article1) - set(common_topics))
    unique_topics_2 = list(set(topics_article2) - set(common_topics))
    
    return {
        "Sentiment Distribution": sentiment_counts,
        "Coverage Differences": comparisons,
        "Topic Overlap": {
            "Common Topics": common_topics,
            "Unique Topics in Article 1": unique_topics_1,
            "Unique Topics in Article 2": unique_topics_2
        }
    }


def generate_tts(text: str, lang: str = 'hi') -> str:
    """
    Generate a Hindi TTS audio file from the given text.
    
    Args:
        text (str): The text to convert.
        lang (str): Language code (default 'hi' for Hindi).
    
    Returns:
        str: Filename of the generated audio.
    """
    try:
        tts = gTTS(text=text, lang=lang)
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        tts.save(filename)
        logging.info(f"TTS generated and saved as {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error generating TTS: {e}")
        return ""
