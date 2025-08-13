#!/usr/bin/env python3
"""
Aurafy (Spotify + SBERT reranker)

- Candidate gathering: artists-by-genre top tracks, genre playlists, targeted track search.
- SBERT (sentence-transformers) reranks semantically to the user's prompt.
- Optional strict-genre filtering (using artist genres).
- Optional Spotify audio-features (if reachable) to add light tags for reranking.

ENV:
  SPOTIFY_CLIENT_ID=...
  SPOTIFY_CLIENT_SECRET=...
  [optional] SBERT_MODEL_NAME=all-MiniLM-L6-v2

Install:
  pip install httpx python-dotenv sentence-transformers torch
"""

import os
import time
import base64
import argparse
import json
import random
import re
from typing import Dict, List, Tuple, Optional, Set

import httpx
from dotenv import load_dotenv

# =========================== Config & Globals ===========================
load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")

DEFAULT_MARKET = "US"
SEARCH_PAGES = 2
YEAR_HINT = "2010-2025"
MAX_CANDIDATES = 400
AUDIO_FEATURES_BATCH = 80
HTTP_TIMEOUT = 30.0
AUDIO_FEATURES_AVAILABLE = None  # tri-state: None/True/False

SBERT_MODEL_NAME_ENV_DEFAULT = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")

_SPOTIFY_ID_RE = re.compile(r"^[A-Za-z0-9]{22}$")
_artist_genres_cache: dict[str, list[str]] = {}

# Local allowlist for recognizable genre tokens
ALLOWED_GENRES: Set[str] = {
    "acoustic","afrobeat","alt-rock","alternative","ambient","anime","bluegrass","blues",
    "bossa-nova","brazil","breakbeat","british","cabaret","cantopop","chicago-house","children",
    "chill","classical","club","comedy","country","dance","dancehall","death-metal","deep-house",
    "detroit-techno","disco","disney","drum-and-bass","dub","dubstep","edm","electro","electronic",
    "emo","folk","forro","french","funk","garage","german","goth","grindcore","groove","grunge",
    "guitar","hard-rock","hardcore","hardstyle","heavy-metal","hip-hop","holidays","honky-tonk",
    "house","idm","indie","indie-pop","industrial","iranian","j-dance","j-idol","j-pop","j-rock",
    "jazz","k-pop","kids","latin","latino","malay","mandopop","metal","metalcore","minimal-techno",
    "movies","mpb","new-age","opera","pagode","party","piano","pop","pop-film","post-dubstep",
    "power-pop","progressive-house","psych-rock","punk","punk-rock","r-n-b","rainy-day","reggae",
    "reggaeton","rock","rock-n-roll","rockabilly","romance","sad","salsa","samba","sertanejo",
    "show-tunes","singer-songwriter","ska","sleep","songwriter","soul","soundtracks","spanish",
    "study","summer","synth-pop","tango","techno","trance","trip-hop","turkish","work-out","world-music"
}

# Optional mood presets; we only use these lightly (tags) when audio features are available
PRESETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "happy":      {"valence": (0.7, 1.0), "energy": (0.6, 0.95), "danceability": (0.6, 0.95), "tempo": (105, 160)},
    "sad":        {"valence": (0.0, 0.35), "energy": (0.2, 0.6),  "acousticness": (0.4, 1.0),  "tempo": (60, 110)},
    "chill":      {"valence": (0.3, 0.7), "energy": (0.2, 0.5),  "acousticness": (0.3, 0.9),  "tempo": (60, 105)},
    "hype":       {"valence": (0.5, 1.0), "energy": (0.8, 1.0),  "danceability": (0.7, 1.0),  "tempo": (120, 180)},
    "nostalgic":  {"valence": (0.4, 0.75),"energy": (0.3, 0.65), "acousticness": (0.2, 0.8),  "tempo": (70, 120)},
}

# =========================== HTTP / Auth ===========================
_http: Optional[httpx.Client] = None
_token_cache = {"access_token": None, "expires_at": 0.0}

def http_client() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(
            timeout=HTTP_TIMEOUT,
            follow_redirects=False,
            http2=False,
            headers={"User-Agent": "aurafy/2.1"}
        )
    return _http

def get_app_token() -> str:
    now = time.time()
    if _token_cache["access_token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["access_token"]
    basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    resp = http_client().post("https://accounts.spotify.com/api/token", data={"grant_type": "client_credentials"}, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = now + float(data.get("expires_in", 3600))
    return _token_cache["access_token"]

def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {get_app_token()}", "Accept": "application/json"}

def _get_json(url: str, params: Optional[dict] = None, *, max_retries: int = 6) -> dict:
    client = http_client()
    backoff = 1.0
    refreshed = False
    last_exc = None
    for _ in range(max_retries):
        try:
            r = client.get(url, params=params, headers=_auth_headers())
            if 200 <= r.status_code < 300:
                return r.json()
            if r.status_code == 401 and not refreshed:
                _token_cache["access_token"] = None
                _token_cache["expires_at"] = 0.0
                refreshed = True
                continue
            if r.status_code == 429:
                delay = float(r.headers.get("Retry-After") or backoff)
                time.sleep(delay)
                backoff = min(backoff * 2, 16) + random.random()
                continue
            if 500 <= r.status_code < 600:
                time.sleep(backoff + random.random())
                backoff = min(backoff * 2, 16)
                continue
            r.raise_for_status()
        except httpx.HTTPError as e:
            last_exc = e
            time.sleep(backoff + random.random())
            backoff = min(backoff * 2, 16)
    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to fetch JSON")

# =========================== Parsing ===========================
def parse_query(text: str) -> tuple[List[str], List[str]]:
    words = [w.strip().lower() for w in text.split() if w.strip()]
    moods: List[str] = []
    genres: List[str] = []
    for w in words:
        if w in PRESETS and w not in moods:
            moods.append(w); continue
        g = w.replace(" ", "-")
        if g in ALLOWED_GENRES and g not in genres:
            genres.append(g)
    return moods, genres

# =========================== Candidate Gathering ===========================
def _clean_track_ids(ids):
    return [tid for tid in ids if tid and _SPOTIFY_ID_RE.match(tid)]

def search_tracks_by_term(term: str, pages: int = SEARCH_PAGES, market: str = DEFAULT_MARKET) -> List[dict]:
    items: List[dict] = []
    queries = [f'track:"" {term} year:{YEAR_HINT}', f'{term} year:{YEAR_HINT}', term]
    seen_ids = set()
    for q in queries:
        for i in range(pages):
            params = {"q": q, "type": "track", "limit": 50, "offset": i * 50, "market": market}
            data = _get_json("https://api.spotify.com/v1/search", params=params)
            tracks = data.get("tracks", {}).get("items", [])
            if not tracks:
                break
            for t in tracks:
                tid = t.get("id")
                if tid and tid not in seen_ids:
                    items.append(t); seen_ids.add(tid)
            time.sleep(0.1)
        if items:
            break
    return items

def search_artists_by_genre(genre: str, limit: int = 20) -> List[dict]:
    queries = [f'genre:"{genre}"', f'genre:{genre}', genre]
    seen = set(); out: List[dict] = []
    for q in queries:
        data = _get_json("https://api.spotify.com/v1/search", {"q": q, "type": "artist", "limit": limit})
        for a in data.get("artists", {}).get("items", []) or []:
            aid = a.get("id")
            if aid and aid not in seen:
                out.append(a); seen.add(aid)
        if out:
            break
    return out[:limit]

def get_artist_top_tracks(artist_id: str, country: str = DEFAULT_MARKET) -> List[dict]:
    data = _get_json(f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks", {"market": country})
    return data.get("tracks", []) or []

def search_tracks_by_genres(genres: List[str], limit_per_genre=160) -> List[dict]:
    all_tracks: List[dict] = []; seen = set()
    for genre in genres[:3]:
        artists = search_artists_by_genre(genre, limit=20)
        for a in artists:
            for t in get_artist_top_tracks(a.get("id")):
                tid = t.get("id")
                if tid and tid not in seen:
                    all_tracks.append(t); seen.add(tid)
            if len(all_tracks) >= limit_per_genre:
                break
        if len(all_tracks) >= limit_per_genre:
            break
    return all_tracks[:limit_per_genre]

def collect_from_playlists(term: str, max_playlists: int = 3, max_tracks_per: int = 150) -> list[dict]:
    data = _get_json("https://api.spotify.com/v1/search", {"q": term, "type": "playlist", "limit": max_playlists})
    pl_items = (data.get("playlists") or {}).get("items") or []
    out: list[dict] = []; seen: set[str] = set()

    def _as_track(obj: dict | None) -> dict | None:
        if not obj or not isinstance(obj, dict): return None
        if obj.get("type") != "track": return None
        if not obj.get("id"): return None
        return obj

    for pl in pl_items:
        if not pl or not isinstance(pl, dict): continue
        pid = pl.get("id")
        if not pid: continue
        pulled = 0; offset = 0
        while pulled < max_tracks_per:
            resp = _get_json(f"https://api.spotify.com/v1/playlists/{pid}/tracks",
                             {"market": DEFAULT_MARKET, "limit": 100, "offset": offset})
            items = resp.get("items") or []
            if not items: break
            for it in items:
                t = _as_track(it.get("track"))
                if not t: continue
                tid = t["id"]
                if tid not in seen:
                    out.append(t); seen.add(tid); pulled += 1
                    if pulled >= max_tracks_per: break
            if len(items) < 100 or pulled >= max_tracks_per: break
            offset += 100
            time.sleep(0.1)

    return out

# =========================== Audio Features (optional) ===========================
def audio_features_available() -> bool:
    global AUDIO_FEATURES_AVAILABLE
    if AUDIO_FEATURES_AVAILABLE is not None:
        return AUDIO_FEATURES_AVAILABLE
    tid = "0VjIjW4GlUZAMYd2vXMi3b"  # Blinding Lights
    r = http_client().get(f"https://api.spotify.com/v1/audio-features/{tid}", headers=_auth_headers())
    AUDIO_FEATURES_AVAILABLE = (r.status_code == 200)
    print(f"[debug] audio_features_available={AUDIO_FEATURES_AVAILABLE}")
    return AUDIO_FEATURES_AVAILABLE

def _get_audio_features_batch(ids_chunk: List[str]) -> Dict[str, dict]:
    feats: Dict[str, dict] = {}
    if not ids_chunk: return feats
    data = _get_json("https://api.spotify.com/v1/audio-features", params={"ids": ",".join(ids_chunk)})
    for f in (data.get("audio_features") or []):
        if f and f.get("id"):
            feats[f["id"]] = f
    return feats

def _get_audio_features_single(tid: str) -> Optional[dict]:
    data = _get_json(f"https://api.spotify.com/v1/audio-features/{tid}")
    if data and data.get("id"): return data
    return None

def get_audio_features(track_ids: List[str]) -> Dict[str, dict]:
    ids = _clean_track_ids(track_ids)
    feats: Dict[str, dict] = {}
    remaining: set[str] = set(ids)
    for i in range(0, len(ids), AUDIO_FEATURES_BATCH):
        chunk = ids[i:i + AUDIO_FEATURES_BATCH]
        got = _get_audio_features_batch(chunk)
        feats.update(got); remaining -= set(got.keys())
        time.sleep(0.05)
    for tid in list(remaining):
        try:
            f = _get_audio_features_single(tid)
        except httpx.HTTPError:
            f = None
        if f: feats[tid] = f
    print(f"[debug] with_features={len(feats)}")
    return feats

# =========================== Genres (strict filtering) ===========================
def get_artist_genres(aid: str) -> list[str]:
    if not aid: return []
    if aid in _artist_genres_cache: return _artist_genres_cache[aid]
    try:
        data = _get_json(f"https://api.spotify.com/v1/artists/{aid}")
        gens = data.get("genres") or []
    except Exception:
        gens = []
    _artist_genres_cache[aid] = gens
    return gens

def artist_matches_any_genre(artist_id: str, targets: list[str]) -> bool:
    gens = [g.lower() for g in get_artist_genres(artist_id)]
    tgt = [t.lower() for t in targets]
    return any(any(t in g for t in tgt) for g in gens)

# =========================== SBERT Re-ranker ===========================
def get_sbert_model(name: str):
    """
    Cached loader for sentence-transformers; no global mutation.
    """
    # simple singleton per process; could extend to multi-name cache if needed
    if not hasattr(get_sbert_model, "_cache"):
        get_sbert_model._cache = {}
    cache = get_sbert_model._cache
    if name in cache:
        return cache[name]
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[debug] Loading SBERT model: {name}")
        model = SentenceTransformer(name)
        cache[name] = model
        return model
    except Exception as e:
        print(f"[warn] sentence-transformers not available: {e}")
        cache[name] = None
        return None

def build_track_text(t: dict, feat: Optional[dict], target_genres: List[str], mood_words: List[str]) -> str:
    name = t.get("name") or ""
    artists = ", ".join(a.get("name","") for a in (t.get("artists") or []))
    agens = []
    for a in (t.get("artists") or []):
        agens.extend(get_artist_genres(a.get("id")))
    agens = list(dict.fromkeys([g.lower() for g in agens]))[:8]
    tags = []
    if feat:
        v = feat.get("valence"); e = feat.get("energy"); d = feat.get("danceability"); ac = feat.get("acousticness"); tp = feat.get("tempo")
        if v is not None: tags.append(f"valence:{v:.2f}")
        if e is not None: tags.append(f"energy:{e:.2f}")
        if d is not None: tags.append(f"danceability:{d:.2f}")
        if ac is not None: tags.append(f"acousticness:{ac:.2f}")
        if tp is not None: tags.append(f"tempo:{int(tp)}bpm")
    return " | ".join(filter(None, [
        f"{name} — {artists}",
        ("genres: " + ", ".join(agens)) if agens else "",
        ("moods: " + ", ".join(mood_words)) if mood_words else "",
        ("targets: " + ", ".join(target_genres)) if target_genres else "",
        ("tags: " + ", ".join(tags)) if tags else ""
    ]))

def sbert_rerank(query: str, tracks: List[dict], feat_map: Dict[str, dict], target_genres: List[str], mood_words: List[str], limit: int, model_name: str) -> List[dict]:
    model = get_sbert_model(model_name)
    if model is None:
        # Fallback: popularity sort
        return sorted(tracks, key=lambda x: (x.get("popularity") or 0), reverse=True)[:limit]

    # Corpus strings
    corpus_texts = [build_track_text(t, feat_map.get(t.get("id")), target_genres, mood_words) for t in tracks]
    try:
        import numpy as np
        q_emb = model.encode([query], normalize_embeddings=True)
        c_emb = model.encode(corpus_texts, normalize_embeddings=True, batch_size=64, convert_to_numpy=True)
        sims = (c_emb @ q_emb[0])  # cosine since normalized

        # Small popularity + genre + feature bonuses
        pop = np.array([float(t.get("popularity") or 0)/100.0 for t in tracks])
        gbonus = np.array([
            0.10 if any(artist_matches_any_genre(a.get("id"), target_genres) for a in (t.get("artists") or [])) else 0.0
            for t in tracks
        ])
        fbonus = []
        for t in tracks:
            af = feat_map.get(t.get("id")) or {}
            v = af.get("valence"); e = af.get("energy"); d = af.get("danceability")
            bonus = 0.0
            if "happy" in mood_words and v is not None and e is not None and v >= 0.65 and e >= 0.6: bonus += 0.12
            if "sad" in mood_words and v is not None and v <= 0.3: bonus += 0.12
            if "chill" in mood_words and e is not None and e <= 0.45: bonus += 0.10
            if "hype" in mood_words and d is not None and e is not None and d >= 0.7 and e >= 0.75: bonus += 0.12
            fbonus.append(bonus)
        fbonus = np.array(fbonus)

        score = 0.75*sims + 0.15*pop + 0.05*gbonus + 0.05*fbonus
        order = np.argsort(-score)[:limit]
        return [tracks[i] for i in order.tolist()]
    except Exception as e:
        print(f"[warn] SBERT rerank failed: {e}")
        return sorted(tracks, key=lambda x: (x.get("popularity") or 0), reverse=True)[:limit]

# =========================== Core Recommend ===========================
def recommend_from_text(text: str, limit: int, *, strict_genre: bool, sbert_model_name: str) -> List[dict]:
    # Parse prompt into mood words + genre tokens (when present)
    moods, genres = parse_query(text)
    if not genres:
        for g in ["pop","indie-pop","dance","rock","edm","hip-hop"]:
            if g in ALLOWED_GENRES:
                genres = [g]; break
        if not genres: genres = ["pop"]

    # Build candidate pool
    candidates = search_tracks_by_genres(genres, limit_per_genre=160)
    for g in genres[:2]:
        candidates.extend(collect_from_playlists(g, max_playlists=3, max_tracks_per=150))
    for g in genres[:2]:
        candidates.extend(search_tracks_by_term(g, pages=1, market=DEFAULT_MARKET))

    # Dedup & cap
    seen = set(); uniq_candidates = []
    for t in candidates:
        tid = t.get("id")
        if tid and tid not in seen:
            uniq_candidates.append(t); seen.add(tid)
    if len(uniq_candidates) > MAX_CANDIDATES:
        uniq_candidates = uniq_candidates[:MAX_CANDIDATES]

    # Strict genre (by artist genres)
    if strict_genre and genres:
        gtokens = genres[:]
        filtered = []
        seen_ids = set()
        for t in uniq_candidates:
            arts = t.get("artists") or []
            if any(artist_matches_any_genre(a.get("id"), gtokens) for a in arts):
                tid = t.get("id")
                if tid and tid not in seen_ids:
                    filtered.append(t); seen_ids.add(tid)
        if filtered:
            uniq_candidates = filtered

    print(f"[debug] candidates={len(candidates)}, unique={len(uniq_candidates)}")
    if not uniq_candidates:
        return []

    # Optional features (for tags only)
    feat_map: Dict[str, dict] = {}
    if audio_features_available():
        ids = [t["id"] for t in uniq_candidates]
        feat_map = get_audio_features(ids)

    # SBERT rerank
    reranked = sbert_rerank(text, uniq_candidates, feat_map, genres, moods, limit=max(limit*2, 30), model_name=sbert_model_name)
    return format_tracks(reranked[:limit])

# =========================== Formatting & CLI ===========================
def format_tracks(tracks: List[dict]) -> List[dict]:
    out = []
    for t in tracks:
        images = t.get("album", {}).get("images", [])
        album_art = images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)
        out.append({
            "id": t.get("id"),
            "name": t.get("name"),
            "artist": ", ".join(a["name"] for a in t.get("artists", [])),
            "album": t.get("album", {}).get("name"),
            "albumArt": album_art,
            "previewUrl": t.get("preview_url") or t.get("previewUrl"),
            "uri": t.get("uri"),
            "popularity": t.get("popularity"),
        })
    return out

def main():
    parser = argparse.ArgumentParser(description="Aurafy (Spotify + SBERT reranker)")
    parser.add_argument("text", nargs="*", help="Anything: 'rainy late night nostalgic drive', 'sad disney', 'happy pop' ...")
    parser.add_argument("--limit", type=int, default=12, help="Number of tracks to return")
    parser.add_argument("--strict-genre", action="store_true", help="Require artist genres to include input genre tokens")
    parser.add_argument("--sbert-model", type=str, default=None, help="sentence-transformers model name (overrides env)")
    args = parser.parse_args()

    model_name = args.sbert_model or SBERT_MODEL_NAME_ENV_DEFAULT
    query = " ".join(args.text).strip() or "happy pop"

    tracks = recommend_from_text(query, limit=args.limit, strict_genre=args.strict_genre, sbert_model_name=model_name)

    print("\nMood2Music · Preview")
    print("Input:", query)
    print("Top Tracks:")
    for i, t in enumerate(tracks, 1):
        line = f"{i:2d}. {t['artist']} — {t['name']}"
        if t.get("previewUrl"):
            line += f"  [preview: {t['previewUrl']}]"
        print(line)

if __name__ == "__main__":
    main()
