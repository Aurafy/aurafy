from flask import Flask, request, jsonify, send_from_directory
import os, logging, traceback
logging.basicConfig(level=logging.INFO)

REQUIRED_ENV = [
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET"
]


app = Flask(__name__, static_folder="dist", static_url_path="")

def check_env():
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing env: {', '.join(missing)}")
# Serve static files from Vite build directory (already committed to dist/)
@app.errorhandler(Exception)
def _err(e):
    app.logger.error("Unhandled error: %s\n%s", e, traceback.format_exc())
    return {"error": str(e)}, 500

# Health check (no heavy imports)
@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/api/ping")
def ping():
    return "pong", 200

import logging
logging.basicConfig(level=logging.INFO)

@app.before_request
def _log_req():
    app.logger.info(f"{request.method} {request.path} args={dict(request.args)}")

# --- API route (lazy import of main) ---
@app.route("/api/recommend")
def recommend():
    import main  # lazy import so startup is fast
    check_env()
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
