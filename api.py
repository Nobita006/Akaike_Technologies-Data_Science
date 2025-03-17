from flask import Flask, request, jsonify, send_file
from utils import fetch_news, process_article, comparative_analysis, generate_tts

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    company = data.get("company")
    if not company:
        return jsonify({"error": "Company name not provided"}), 400
    
    # Fetch news articles using the improved function
    raw_articles = fetch_news(company)
    if not raw_articles:
        return jsonify({"error": f"No articles found for {company}."}), 404

    processed_articles = [process_article(article) for article in raw_articles]
    
    # Comparative sentiment analysis
    comp_analysis = comparative_analysis(processed_articles)
    
    # Aggregate final sentiment (simple majority count)
    pos = sum(1 for art in processed_articles if art["Sentiment"] == "Positive")
    neg = sum(1 for art in processed_articles if art["Sentiment"] == "Negative")
    
    if pos > neg:
        final_sentiment = f"{company}'s news coverage is mostly positive."
    elif neg > pos:
        final_sentiment = f"{company}'s news coverage is mostly negative."
    else:
        final_sentiment = "Overall, the news coverage seems balanced."
    
    # Generate Hindi TTS for the final sentiment summary
    audio_text = f"Company {company}. {final_sentiment}"
    audio_file = generate_tts(audio_text, lang='hi')
    
    response = {
        "Company": company,
        "Articles": processed_articles,
        "Comparative Sentiment Score": comp_analysis,
        "Final Sentiment Analysis": final_sentiment,
        "Audio": audio_file  # Filename; accessible via /audio endpoint
    }
    
    return jsonify(response)

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    try:
        return send_file(filename, mimetype="audio/mp3")
    except Exception as e:
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
