from flask import Flask, request, jsonify, after_this_request, make_response
from utils import (
    fetch_news, process_article, comparative_analysis, generate_tts,
    advanced_summarize, extended_analysis, build_embeddings
)
import os

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    print("Home endpoint accessed.")
    return {"status": "Backend is running!"}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    company = data.get("company")
    if not company:
        return jsonify({"error": "Company name not provided"}), 400

    print(f"Analyzing company: {company}")

    # Fetch BBC articles asynchronously
    raw_articles = fetch_news(company, num_articles=15)
    if not raw_articles:
        return jsonify({"error": f"No articles found for '{company}'."}), 404

    processed_articles = [process_article(a) for a in raw_articles]

    # Minimal output
    articles_output = [{
        "Title": art["Title"],
        "Summary": art["Summary"],
        "Sentiment": art["Sentiment"],
        "Topics": art["Topics"]
    } for art in processed_articles]

    # Basic + Extended Analysis
    comp_analysis = comparative_analysis(processed_articles)
    ext_analysis = extended_analysis(processed_articles)
    embeddings = build_embeddings(processed_articles)

    # Determine majority sentiment
    sentiment_dist = comp_analysis.get("Sentiment Distribution", {})
    majority = "Neutral"
    pos_count = sentiment_dist.get("Positive", 0)
    neg_count = sentiment_dist.get("Negative", 0)
    if pos_count > neg_count:
        majority = "Positive"
    elif neg_count > pos_count:
        majority = "Negative"

    # Summarize all articles
    aggregated_text = " ".join(a["Summary"] for a in processed_articles)
    short_summary = advanced_summarize(aggregated_text, num_sentences=1)
    final_sentiment = f"{company}'s latest news is mostly {majority}. {short_summary}"
    final_sentiment = final_sentiment.replace(" .", ".")

    # Generate TTS audio
    audio_file = generate_tts(f"कंपनी {company}. {final_sentiment}", lang='hi')
    print(f"TTS generated: {audio_file}")

    return jsonify({
        "Company": company,
        "Articles": articles_output,
        "Comparative Sentiment Score": comp_analysis,
        "Extended Analysis": ext_analysis,
        "Final Sentiment Analysis": final_sentiment,
        "Embeddings": embeddings,  # for semantic search
        "Audio": audio_file
    })

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    print(f"Serving audio file: {filename}")
    @after_this_request
    def remove_file(response):
        try:
            os.remove(filename)
            print(f"Removed file: {filename}")
        except Exception as error:
            print(f"Error removing file {filename}: {error}")
        return response

    try:
        with open(filename, "rb") as f:
            audio_data = f.read()
        response = make_response(audio_data)
        response.headers["Content-Type"] = "audio/mp3"
        return response
    except Exception as e:
        print(f"Error reading audio file {filename}: {e}")
        return jsonify({"error": "Audio file not found"}), 404

if __name__ == '__main__':
    # Disable the reloader to avoid duplicate resource loading in debug mode.
    app.run(debug=True, use_reloader=False)
