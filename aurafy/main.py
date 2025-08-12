import os, time, base64, argparse, json
import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")

# ---- simple presets (replace with LLM later) ----
PRESETS = {
    "happy": {"valence": (0.7, 1.0), "energy": (0.6, 0.9), "danceability": (0.6, 0.9), "tempo": (110, 140)},
    "nostalgic": {"valence": (0.4, 0.7), "energy": (0.3, 0.6), "acousticness": (0.4, 0.9), "tempo": (70, 110)},
    "sad": {"valence": (0.0, 0.3), "energy": (0.2, 0.5), "acousticness": (0.5, 1.0), "tempo": (60, 90)},
    "hype": {"valence": (0.5, 0.9), "energy": (0.8, 1.0), "danceability": (0.7, 1.0), "tempo": (120, 160)},
    "chill": {"valence": (0.3, 0.7), "energy": (0.2, 0.5), "acousticness": (0.3, 0.9), "tempo": (60, 100)},
}

ALLOWED_SEEDS_CACHE = {"genres": None, "fetched_at": 0}
def analyze_text(text: str):
    t = text.lower()
    tags = []
    features = {}

    for key in PRESETS:
        if key in t:
            tags.append(key)
            # merge ranges
            for k, rng in PRESETS[key].items():
                if k not in features:
                    features[k] = rng
                else:
                    a, b = features[k], rng
                    features[k] = (min(a[0], b[0]), max(a[1], b[1]))

    # fallback: infer from adjectives / defaults
    if not tags:
        # lightweight heuristics
        if any(w in t for w in ["rain", "late night", "alone", "nostalg"]):
            tags = ["nostalgic", "chill"]
            features = PRESETS["nostalgic"]
        elif any(w in t for w in ["gym", "run", "party", "hype"]):
            tags = ["hype"]
            features = PRESETS["hype"]
        elif any(w in t for w in ["sad", "blue", "low"]):
            tags = ["sad"]
            features = PRESETS["sad"]
        else:
            tags = ["chill"]
            features = PRESETS["chill"]

    return tags, features

def get_allowed_genres():
    # cache for 1 hour
    if ALLOWED_SEEDS_CACHE["genres"] and time.time() - ALLOWED_SEEDS_CACHE["fetched_at"] < 3600:
        return ALLOWED_SEEDS_CACHE["genres"]
    data = spotify_get("https://api.spotify.com/v1/recommendations/available-genre-seeds")
    genres = set(data.get("genres", []))
    ALLOWED_SEEDS_CACHE["genres"] = genres
    ALLOWED_SEEDS_CACHE["fetched_at"] = time.time()
    return genres

_token_cache = {"access_token": None, "expires_at": 0}

def get_app_token() -> str:
    now = time.time()
    if _token_cache["access_token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["access_token"]

    basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    resp = httpx.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = now + int(data["expires_in"])
    return _token_cache["access_token"]

def spotify_get(url: str, params: dict | None = None):
    token = get_app_token()
    headers = {"Authorization": f"Bearer {token}"}
    r = httpx.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()

def pick_seed_genres(tags: list[str]) -> str:
    allowed = get_allowed_genres()
    # normalize tags (spaces -> hyphens)
    norm = []
    for t in tags:
        t = t.lower().strip().replace(" ", "-")
        if t in allowed:
            norm.append(t)
    # fallback if none match
    if not norm:
        # choose a few safe defaults known to exist
        defaults = ["pop", "rock", "indie", "hip-hop", "chill"]
        norm = [g for g in defaults if g in allowed][:5]
        if not norm:  # ultra fallback
            norm = ["pop"]
    # Spotify allows up to 5 seeds total (across artists/tracks/genres)
    return ",".join(norm[:5])

def spotify_get_verbose(url: str, params: dict | None = None):
    try:
        return spotify_get(url, params)
    except httpx.HTTPStatusError as e:
        # Helpful debug info
        body = e.response.text if e.response is not None else "<no body>"
        raise SystemExit(f"Spotify error {e.response.status_code} for {e.request.url}\n{body}")

def recommend(tags: list[str], features: dict, limit: int = 20):
    seed_genres = pick_seed_genres(tags)
    params: dict = {
        "limit": limit,
        "market": "US",             # helps in some edge cases
        "seed_genres": seed_genres, # validated against Spotify's list
    }

    mapping = {
        "valence": ("min_valence", "max_valence"),
        "energy": ("min_energy", "max_energy"),
        "danceability": ("min_danceability", "max_danceability"),
        "acousticness": ("min_acousticness", "max_acousticness"),
        "tempo": ("min_tempo", "max_tempo"),
    }
    for k, (mn, mx) in mapping.items():
        if k in features and features[k]:
            a, b = features[k]
            params[mn], params[mx] = a, b

    data = spotify_get_verbose("https://api.spotify.com/v1/recommendations", params=params)
    out = []
    for t in data.get("tracks", []):
        out.append({
            "id": t["id"],
            "name": t["name"],
            "artist": ", ".join(a["name"] for a in t["artists"]),
            "album": t["album"]["name"],
            "albumArt": (t["album"]["images"][1]["url"] if t["album"]["images"] else None),
            "previewUrl": t.get("preview_url"),
            "uri": t.get("uri"),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Mood2Music CLI")
    parser.add_argument("text", nargs="*", help="Mood text, e.g. 'rainy nostalgic late-night drive'")
    parser.add_argument("--limit", type=int, default=12, help="Number of tracks")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of pretty text")
    args = parser.parse_args()

    text = " ".join(args.text).strip()
    if not text:
        text = "nostalgic rainy late-night drive"

    tags, features = analyze_text(text)
    tracks = recommend(tags, features, limit=args.limit)

    if args.json:
        print(json.dumps({"input": text, "tags": tags, "features": features, "tracks": tracks}, indent=2))
        return

    print("\nMood2Music · Preview")
    print("Input:", text)
    print("Tags:", ", ".join(tags))
    pretty_feats = ", ".join(f"{k}={v[0]:.2f}–{v[1]:.2f}" for k, v in features.items())
    print("Features:", pretty_feats)
    print("\nTop Tracks:")
    for i, t in enumerate(tracks, 1):
        line = f"{i:2d}. {t['artist']} — {t['name']}"
        if t["previewUrl"]:
            line += f"  [preview: {t['previewUrl']}]"
        print(line)

if __name__ == "__main__":
    main()
