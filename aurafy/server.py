from flask import Flask, request, jsonify, send_from_directory
import os

# Serve static files from Vite build directory (already committed to dist/)
app = Flask(__name__, static_folder="dist", static_url_path="")

# Health check (no heavy imports)
@app.route("/healthz")
def healthz():
    return "ok", 200

# --- API route (lazy import of main) ---
@app.route("/api/recommend")
def recommend():
    import main  # lazy import so startup is fast
    text = request.args.get("text", "")
    limit = int(request.args.get("limit", 12))
    tracks = main.recommend(text, limit)
    return jsonify(tracks)

# --- Frontend routes ---
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    file_path = os.path.join(app.static_folder, path)
    if os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    # local dev only
    app.run(host="0.0.0.0", port=5000)
