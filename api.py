from flask import Flask, request, jsonify, send_file
from utils import fetch_news, process_article, comparative_analysis, generate_tts, advanced_summarize

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    company = data.get("company")
    if not company:
        return jsonify({"error": "Company name not provided"}), 400
    
    # Fetch exactly 15 BBC articles for the given company
    raw_articles = fetch_news(company, num_articles=15)
    if not raw_articles:
        return jsonify({"error": f"No articles found for '{company}'."}), 404
    
    # Process each article: advanced summarization, sentiment analysis, topic extraction
    processed_articles = [process_article(article) for article in raw_articles]
    
    # Prepare final output for articles (only Title, Summary, Sentiment, and Topics)
    articles_output = [{
        "Title": art.get("Title"),
        "Summary": art.get("Summary"),
        "Sentiment": art.get("Sentiment"),
        "Topics": art.get("Topics")
    } for art in processed_articles]
    
    # Compute overall comparative analysis
    comp_analysis = comparative_analysis(processed_articles)
    sentiment_distribution = comp_analysis.get("Sentiment Distribution", {})
    
    # Determine majority sentiment across all articles
    majority = "Neutral"
    if sentiment_distribution.get("Positive", 0) > sentiment_distribution.get("Negative", 0):
        majority = "Positive"
    elif sentiment_distribution.get("Negative", 0) > sentiment_distribution.get("Positive", 0):
        majority = "Negative"
    
    # Aggregate all article summaries and generate a short dynamic summary (one sentence)
    aggregated_text = " ".join([art["Summary"] for art in processed_articles])
    short_summary = advanced_summarize(aggregated_text, num_sentences=1)
    
    # Form a concise final sentiment analysis using the company name, majority sentiment, and short summary.
    final_sentiment = f"{company}'s latest news is mostly {majority}. {short_summary}"
    final_sentiment = final_sentiment.replace(" .", ".")
    
    # Generate Hindi TTS for the final sentiment analysis
    audio_text = f"कंपनी {company}. {final_sentiment}"
    audio_file = generate_tts(audio_text, lang='hi')
    
    response = {
        "Company": company,
        "Articles": articles_output,
        "Comparative Sentiment Score": comp_analysis,
        "Final Sentiment Analysis": final_sentiment,
        "Audio": audio_file
    }
    
    return jsonify(response)

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    try:
        return send_file(filename, mimetype="audio/mp3")
    except Exception:
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
