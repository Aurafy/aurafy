from flask import Flask, request, jsonify
import main

app = Flask(__name__)

@app.route("/api/recommend")
def recommend():
    text = request.args.get("text", "")
    limit = int(request.args.get("limit", 12))

    tracks = main.recommend(text, limit)  # now works ✅
    return jsonify(tracks)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
