# server.py
from flask import Flask, request, jsonify, send_from_directory
import os
import main

# Serve static files from Vite build directory
app = Flask(__name__, static_folder="dist", static_url_path="")

# --- API routes ---
@app.route("/api/recommend")
def recommend():
    text = request.args.get("text", "")
    limit = int(request.args.get("limit", 12))
    tracks = main.recommend(text, limit)
    return jsonify(tracks)

# --- Frontend routes ---
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# Serve any file in dist (e.g. /assets/...)
@app.route("/<path:path>")
def static_proxy(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    # SPA fallback for client-side routes
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
