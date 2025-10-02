from flask import Flask, render_template, request, jsonify
import pickle

# Load trained pipeline (TF-IDF + Model together)
with open("tweet_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# --------- UI Route ---------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    tweet_text = ""

    if request.method == "POST":
        tweet_text = request.form["tweet"]

        if tweet_text.strip() != "":
            # Use pipeline directly
            result = pipeline.predict([tweet_text])[0]
            prediction = "🚨 Disaster" if result == 1 else "🙂 Not a Disaster"

    return render_template("index.html", prediction=prediction, tweet=tweet_text)


# --------- API Route ---------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        tweet = data.get("tweet", "")

        if not tweet.strip():
            return jsonify({"error": "No tweet text provided"}), 400

        result = pipeline.predict([tweet])[0]
        label = "Disaster" if result == 1 else "Non-Disaster"

        return jsonify({
            "tweet": tweet,
            "prediction": int(result),
            "label": label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)



    
    
