#!/usr/bin/env python3
"""
Aurafy (Spotify + SBERT reranker) — optimized, accuracy-tuned, and popularity-mixed.

Highlights:
- File-backed GET cache (TTL) across runs.
- Batched artist metadata via /v1/artists?ids=... (fewer requests).
- Two-phase rerank: semantic shortlist first; fetch audio-features only for shortlist.
- Shallow playlist harvesting (first page only), early stop when pool is "good enough".
- In-process request coalescing to avoid duplicate concurrent GETs.
- Jazz-aware filtering to avoid non-jazz bleed for prompts like "jazzy cafe".
- Optional unique-artist constraint (--unique-artists / --one-per-artist) for cleaner variety.
- Popularity-aware mixing: sprinkle well-known (anchor) songs with discovery via --anchor-ratio.
- Popularity floor (--min-popularity) to keep long-tail out of the pool.
- Post-rerank de-dup (by artist+normalized title) and optional one-per-artist cap.

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
import numpy as np
from sentence_transformers import util
import hashlib
import pathlib
import threading

import httpx
from dotenv import load_dotenv

# =========================== Config & Globals ===========================
load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")

DEFAULT_MARKET = "US"
YEAR_HINT = "2010-2025"
AUDIO_FEATURES_BATCH = 80
HTTP_TIMEOUT = 30.0
AUDIO_FEATURES_AVAILABLE = None  # tri-state: None/True/False

SBERT_MODEL_NAME_ENV_DEFAULT = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")

# Request & search tuning
SEARCH_PAGES = 1                # fewer pages on direct track search
MAX_PLAYLISTS_DEFAULT = 5       # shallower playlist search
MAX_TRACKS_PER_PLAYLIST = 100   # only first page per playlist

# Early stop targets and shortlist sizing
EARLY_POOL_TARGET = 600         # stop gathering once we hit this many unique candidates
SHORTLIST_AFTER_PASS1 = 100     # fetch audio features only for this many top candidates

# File-backed cache for GETs
REQUEST_CACHE_DIR = ".aurafy_cache"
REQUEST_CACHE_TTL = 24 * 3600  # seconds

# In-memory in-process request de-dupe
_inflight_keys: set[str] = set()

_SPOTIFY_ID_RE = re.compile(r"^[A-Za-z0-9]{22}$")

# Bulk artist meta cache
_artist_meta_cache: dict[str, dict] = {}

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

# Optional mood presets; used lightly when audio features are available
PRESETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "happy":      {"valence": (0.7, 1.0), "energy": (0.6, 0.95), "danceability": (0.6, 0.95), "tempo": (105, 160)},
    "sad":        {"valence": (0.0, 0.35), "energy": (0.2, 0.6),  "acousticness": (0.4, 1.0),  "tempo": (60, 110)},
    "chill":      {"valence": (0.3, 0.7), "energy": (0.2, 0.5),  "acousticness": (0.3, 0.9),  "tempo": (60, 105)},
    "hype":       {"valence": (0.5, 1.0), "energy": (0.8, 1.0),  "danceability": (0.7, 1.0),  "tempo": (120, 180)},
    "nostalgic":  {"valence": (0.4, 0.75),"energy": (0.3, 0.65), "acousticness": (0.2, 0.8),  "tempo": (70, 120)},
}

# ===== Popularity-aware mixing (anchors vs discovery) =====
ANCHOR_POP_THRESHOLD = 65          # track popularity >= this is considered an anchor
ANCHOR_ARTIST_POP_THRESHOLD = 70   # or any artist on the track with popularity >= this
ANCHOR_RATIO_DEFAULT = 0.60        # target fraction of anchors in final list
ANCHOR_SPAN = 2                    # aim ~1 anchor every 2 items until target met

# ---- Jazz family helpers ----
JAZZ_FAMILY_KEYWORDS = {
    "jazz", "jazzy", "bossa", "bossa nova", "swing", "bebop", "big band",
    "cool jazz", "hard bop", "latin jazz", "lounge", "nu jazz", "smooth jazz"
}

# If user typed a jazz-y query, exclude these *unless* the artist also has a jazz tag
# (keeps out reggaeton/latino pop that slip in from playlists)
JAZZ_EXCLUDE_ROOTS = {
    "reggaeton", "latino", "trap latino", "regional mexican", "corridos",
    "corrido", "bachata", "cumbia", "sertanejo", "pagode", "salsa", "urbano latino"
}

# =========================== HTTP / Auth + Cache ===========================
_http: Optional[httpx.Client] = None
_token_cache = {"access_token": None, "expires_at": 0.0}

pathlib.Path(REQUEST_CACHE_DIR).mkdir(exist_ok=True)
_cache_lock = threading.Lock()

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
    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    resp = http_client().post("https://accounts.spotify.com/api/token", data={"grant_type": "client_credentials"}, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = now + float(data.get("expires_in", 3600))
    return _token_cache["access_token"]

def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {get_app_token()}", "Accept": "application/json"}

def _cache_key(url: str, params: Optional[dict]) -> str:
    base = url + "?" + json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(base.encode()).hexdigest()

def _cache_path(key: str) -> str:
    return os.path.join(REQUEST_CACHE_DIR, f"{key}.json")

def _cache_get(url: str, params: Optional[dict]) -> Optional[dict]:
    key = _cache_key(url, params)
    p = _cache_path(key)
    try:
        st = os.stat(p)
        if (time.time() - st.st_mtime) > REQUEST_CACHE_TTL:
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None

def _cache_set(url: str, params: Optional[dict], data: dict) -> None:
    key = _cache_key(url, params)
    p = _cache_path(key)
    try:
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, p)
    except Exception:
        pass

def _get_json(url: str, params: Optional[dict] = None, *, max_retries: int = 6, cache_ok: bool = True) -> dict:
    # Cache read
    if cache_ok:
        cached = _cache_get(url, params)
        if cached is not None:
            return cached

    client = http_client()
    backoff = 1.0
    refreshed = False
    last_exc = None
    key = _cache_key(url, params)

    # in-process de-dupe: if identical request already in-flight, wait + recheck cache
    global _inflight_keys
    for _ in range(3):
        with _cache_lock:
            if key in _inflight_keys:
                pass
            else:
                _inflight_keys.add(key); break
        time.sleep(0.15)
        c2 = _cache_get(url, params) if cache_ok else None
        if c2 is not None:
            return c2

    try:
        for _ in range(max_retries):
            try:
                r = client.get(url, params=params, headers=_auth_headers())
                if 200 <= r.status_code < 300:
                    data = r.json()
                    if cache_ok:
                        _cache_set(url, params, data)
                    return data
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
    finally:
        with _cache_lock:
            _inflight_keys.discard(key)

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

def search_tracks_by_term(term: str, pages: int = SEARCH_PAGES, market: str = DEFAULT_MARKET, year_hint: Optional[str] = None) -> List[dict]:
    items: List[dict] = []
    yh = year_hint or YEAR_HINT
    queries = [f'track:"" {term} year:{yh}', f'{term} year:{yh}', term]
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

def collect_from_playlists_with_titles(term: str, max_playlists: int = MAX_PLAYLISTS_DEFAULT, max_tracks_per: int = MAX_TRACKS_PER_PLAYLIST) -> list[tuple[dict, str]]:
    data = _get_json("https://api.spotify.com/v1/search", {"q": term, "type": "playlist", "limit": max_playlists})
    pl_items = (data.get("playlists") or {}).get("items") or []
    out: list[tuple[dict, str]] = []
    seen_track_pl: set[tuple[str, str]] = set()

    def _as_track(obj: dict | None) -> dict | None:
        if not obj or not isinstance(obj, dict): return None
        if obj.get("type") != "track": return None
        if not obj.get("id"): return None
        return obj

    for pl in pl_items:
        if not pl or not isinstance(pl, dict): continue
        pid = pl.get("id"); ptitle = (pl.get("name") or "").strip()
        if not pid: continue
        # single page fetch (limit=100)
        resp = _get_json(f"https://api.spotify.com/v1/playlists/{pid}/tracks",
                         {"market": DEFAULT_MARKET, "limit": min(100, max_tracks_per), "offset": 0})
        items = resp.get("items") or []
        pulled = 0
        for it in items:
            t = _as_track(it.get("track"))
            if not t: continue
            tid = t["id"]
            key = (tid, pid)
            if key not in seen_track_pl:
                out.append((t, ptitle))
                seen_track_pl.add(key)
                pulled += 1
                if pulled >= max_tracks_per: break
    return out

# =========================== Audio Features (optional) ===========================
def audio_features_available() -> bool:
    global AUDIO_FEATURES_AVAILABLE
    if AUDIO_FEATURES_AVAILABLE is not None:
        return AUDIO_FEATURES_AVAILABLE
    try:
        tid = "0VjIjW4GlUZAMYd2vXMi3b"  # known valid ID
        r = http_client().get(f"https://api.spotify.com/v1/audio-features/{tid}", headers=_auth_headers())
        AUDIO_FEATURES_AVAILABLE = (r.status_code == 200)
    except Exception:
        AUDIO_FEATURES_AVAILABLE = False
    print(f"[debug] audio_features_available={AUDIO_FEATURES_AVAILABLE}")
    return AUDIO_FEATURES_AVAILABLE

def _get_audio_features_batch(ids_chunk: List[str]) -> Dict[str, dict]:
    feats: Dict[str, dict] = {}
    if not ids_chunk: return feats
    data = _get_json("https://api.spotify.com/v1/audio-features", params={"ids": ",".join(ids_chunk)})
    for f in (data.get("audio_features") or []):
        if f and f.get("id"):
            feats[f["id"]] = f
    if feats and AUDIO_FEATURES_AVAILABLE is False:
        globals()["AUDIO_FEATURES_AVAILABLE"] = True
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
    for tid in list(remaining):
        try:
            f = _get_audio_features_single(tid)
        except httpx.HTTPError:
            f = None
        if f: feats[tid] = f
    print(f"[debug] with_features={len(feats)}")
    return feats

# =========================== Batched Artist Meta ===========================
def _get_artists_bulk(ids: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    ids = [i for i in ids if i and _SPOTIFY_ID_RE.match(i)]
    for i in range(0, len(ids), 50):
        chunk = ids[i:i+50]
        if not chunk: continue
        data = _get_json("https://api.spotify.com/v1/artists", params={"ids": ",".join(chunk)})
        for a in (data.get("artists") or []):
            if a and a.get("id"):
                out[a["id"]] = a
    return out

def prefetch_artist_meta_for_tracks(tracks: List[dict]) -> None:
    need: List[str] = []
    for t in tracks:
        for a in (t.get("artists") or []):
            aid = a.get("id")
            if aid and aid not in _artist_meta_cache:
                need.append(aid)
    if not need:
        return
    fetched = _get_artists_bulk(list(dict.fromkeys(need)))
    _artist_meta_cache.update(fetched)

def _artist_meta(aid: str) -> dict:
    if not aid: return {}
    if aid in _artist_meta_cache:
        return _artist_meta_cache[aid]
    data = _get_json(f"https://api.spotify.com/v1/artists/{aid}")
    if data and data.get("id"):
        _artist_meta_cache[aid] = data
    return _artist_meta_cache.get(aid, {})

def get_artist_popularity(aid: str) -> int:
    if not aid:
        return 0
    m = _artist_meta(aid)
    try:
        return int(m.get("popularity") or 0)
    except Exception:
        return 0

def get_artist_genres(aid: str) -> list[str]:
    if not aid: return []
    m = _artist_meta(aid)
    return list(m.get("genres") or [])

def artist_matches_any_genre(artist_id: str, targets: list[str]) -> bool:
    gens = [g.lower() for g in get_artist_genres(artist_id)]
    tgt = [t.lower() for t in targets]
    return any(any(t in g for t in tgt) for g in gens)

# Helpers for jazz checks
def _has_any_genre(aid: str, needles: set[str]) -> bool:
    gs = [g.lower() for g in get_artist_genres(aid)]
    return any(any(k in g for k in needles) for g in gs)

def _has_any_root(aid: str, roots: set[str]) -> bool:
    gs = [g.lower() for g in get_artist_genres(aid)]
    return any(any(r in g for r in roots) for g in gs)

# =========================== SBERT Re-ranker ===========================
def get_sbert_model(name: str):
    """
    Cached loader for sentence-transformers; no global mutation.
    """
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

def sbert_rerank(
    query: str,
    tracks: list,
    feat_map: dict,
    target_genres: list,
    mood_words: list,
    limit: int,
    model_name: str,
    freq_counts: Optional[dict] = None,
    title_boosts: Optional[dict] = None,
    *,
    anchor_ratio: float = ANCHOR_RATIO_DEFAULT,
    anchor_pop_threshold: int = ANCHOR_POP_THRESHOLD,
):
    """
    Signals:
      - sims_query: SBERT cosine(query, track_text)
      - tb: SBERT similarity(query, playlist titles containing the track)
      - co: evidence (# of query-matched playlists containing the track)
      - gbonus/pop/jazz: tiny nudges
    """
    freq_counts = freq_counts or {}
    title_boosts = title_boosts or {}

    model = get_sbert_model(model_name)
    if model is None:
        return sorted(tracks, key=lambda x: (x.get("popularity") or 0), reverse=True)[:limit]

    # Build corpus
    def describe_features(af):
        if not af: return ""
        parts = []
        if af.get("valence") is not None:      parts.append(f"valence {af['valence']:.2f}")
        if af.get("energy") is not None:       parts.append(f"energy {af['energy']:.2f}")
        if af.get("danceability") is not None: parts.append(f"danceability {af['danceability']:.2f}")
        if af.get("tempo") is not None:        parts.append(f"tempo {af['tempo']:.0f} BPM")
        return ", ".join(parts)

    corpus_texts = []
    for t in tracks:
        desc = build_track_text(t, feat_map.get(t.get("id")), target_genres, mood_words)
        feat_desc = describe_features(feat_map.get(t.get("id")))
        if feat_desc:
            desc += " | " + feat_desc
        corpus_texts.append(desc)

    # Embeddings
    print(f"[progress] Encoding {len(corpus_texts)} track descriptions with SBERT...")
    track_embs = model.encode(corpus_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    print("[progress] Finished encoding tracks")
    print("[progress] Encoding query...")
    query_emb  = model.encode([query], normalize_embeddings=True)
    print("[progress] Finished encoding query]")
    sims_query = util.cos_sim(query_emb, track_embs).cpu().numpy()[0]  # (N,)

    # Evidence signals
    co = np.array([freq_counts.get(t.get("id"), 0) for t in tracks], dtype=float)
    co = np.log1p(co)
    if co.max() > 0: co = co / co.max()
    else: co = np.zeros(len(tracks))

    tb = np.array([title_boosts.get(t.get("id"), 0.0) for t in tracks], dtype=float)
    if tb.max() > 0: tb = tb / tb.max()
    else: tb = np.zeros(len(tracks))

    pop = np.array([float(t.get("popularity") or 0) / 100.0 for t in tracks])

    # Genre tiny bonus + optional K-pop penalty; optional jazz bonus
    blob_cache: dict[str, str] = {}
    def _genre_blob(t):
        tid = t.get("id")
        if tid in blob_cache:
            return blob_cache[tid]
        gs = []
        for a in (t.get("artists") or []):
            gs.extend(get_artist_genres(a.get("id")))
        s = " ".join(gs).lower()
        blob_cache[tid] = s
        return s

    gbonus = np.array([
        0.08 if any(artist_matches_any_genre(a.get("id"), target_genres) for a in (t.get("artists") or [])) else 0.0
        for t in tracks
    ])

    user_asked_kpop = any(g in ("k-pop", "kpop") for g in target_genres) or ("k-pop" in query.lower() or "kpop" in query.lower())
    kpop_pen = np.array([0.06 if (("k-pop" in _genre_blob(t) or "kpop" in _genre_blob(t)) and not user_asked_kpop) else 0.0 for t in tracks])

    is_jazzy_query = any(k in query.lower() for k in JAZZ_FAMILY_KEYWORDS)
    if is_jazzy_query:
        jazz_bonus = np.array([
            0.05 if any(_has_any_genre(a.get("id"), JAZZ_FAMILY_KEYWORDS) for a in (t.get("artists") or [])) else 0.0
            for t in tracks
        ])
    else:
        jazz_bonus = np.zeros(len(tracks))

    # Final score
    score = (
        0.62 * sims_query +
        0.24 * tb +
        0.08 * co +
        0.04 * gbonus +
        0.02 * pop +
        jazz_bonus -
        kpop_pen
    )

    order = np.argsort(-score).tolist()
    ordered = [tracks[i] for i in order]

    # ---- Anchor/Discovery identification & mixing ----
    def is_anchor_track(t: dict) -> bool:
        # Track popularity check
        if (t.get("popularity") or 0) >= anchor_pop_threshold:
            return True
        # Any artist popularity high?
        for a in (t.get("artists") or []):
            aid = a.get("id")
            if aid and get_artist_popularity(aid) >= ANCHOR_ARTIST_POP_THRESHOLD:
                return True
        return False

    anchors  = [t for t in ordered if is_anchor_track(t)]
    discover = [t for t in ordered if not is_anchor_track(t)]

    min_anchors = max(1, int(round(limit * max(0.0, min(1.0, anchor_ratio)))))
    final: list[dict] = []
    ai = di = 0
    i = 0

    # Interleave to "sprinkle" anchors regularly:
    while len(final) < limit and (ai < len(anchors) or di < len(discover)):
        need_anchor = (len(final) < min_anchors) and (i % ANCHOR_SPAN == 0)
        if need_anchor and ai < len(anchors):
            final.append(anchors[ai]); ai += 1
        elif di < len(discover):
            final.append(discover[di]); di += 1
        elif ai < len(anchors):
            final.append(anchors[ai]); ai += 1
        i += 1

    # Top up if needed (favor discovery first for variety)
    while len(final) < limit and di < len(discover):
        final.append(discover[di]); di += 1
    while len(final) < limit and ai < len(anchors):
        final.append(anchors[ai]); ai += 1

    return final[:limit]

# =========================== Core Recommend ===========================
def recommend_from_text(
    text: str,
    limit: int,
    *,
    strict_genre: bool,
    sbert_model_name: str,
    unique_artists: bool = False,
    anchor_ratio: float = ANCHOR_RATIO_DEFAULT,
    anchor_pop_threshold: int = ANCHOR_POP_THRESHOLD,
    min_popularity: int = 0,
) -> List[dict]:
    moods, genres = parse_query(text)
    if not genres:
        genres = ["pop"]

    # Auto-tighten mainstream pop prompts if user didn't set a floor
    q_low = text.lower()
    if any(k in q_low for k in ["bubblegum pop", "bubblegum", "top pop", "chart pop"]) and min_popularity == 0:
        min_popularity = 55
        anchor_ratio = max(anchor_ratio, 0.7)

    candidates: list[dict] = []
    seen_ids: set[str] = set()

    def _add(tracks: List[dict]):
        nonlocal candidates, seen_ids
        for t in tracks:
            tid = (t or {}).get("id")
            if tid and tid not in seen_ids:
                candidates.append(t); seen_ids.add(tid)

    # a) Genre → artist → top tracks
    seed_genres = genres[:] if genres else ["pop", "indie-pop", "dance", "rock", "edm", "hip-hop"]
    _add(search_tracks_by_genres(seed_genres, limit_per_genre=160))

    if len(seen_ids) < EARLY_POOL_TARGET:
        # b) Playlists (shallow)
        pl_pairs = collect_from_playlists_with_titles(text, max_playlists=MAX_PLAYLISTS_DEFAULT, max_tracks_per=MAX_TRACKS_PER_PLAYLIST)
        from collections import defaultdict, Counter
        title_map: dict[str, list[str]] = defaultdict(list)
        for t, ptitle in pl_pairs:
            _add([t])
            tid = t.get("id")
            if tid and ptitle:
                title_map[tid].append(ptitle)

        # c) Direct track search (reduced pages)
        _add(search_tracks_by_term(text, pages=SEARCH_PAGES, market=DEFAULT_MARKET, year_hint=None))
    else:
        title_map = {}

    print(f"[progress] Candidate pool gathered: {len(candidates)} tracks; unique={len(seen_ids)}")

    uniq_candidates = candidates

    # Detect "jazzy" intent from the query tokens themselves
    is_jazzy_query = (
        any(k in q_low for k in JAZZ_FAMILY_KEYWORDS) or
        any(g in ("jazz", "jazzy") for g in genres)
    )

    # Prefetch once so genre lookups are cheap
    prefetch_artist_meta_for_tracks(uniq_candidates)

    # Strict genre filter if requested
    if strict_genre and genres:
        gtokens = genres[:]
        filtered = []
        f_seen = set()
        for t in uniq_candidates:
            if any(artist_matches_any_genre(a.get("id"), gtokens) for a in (t.get("artists") or [])):
                tid = t.get("id")
                if tid and tid not in f_seen:
                    filtered.append(t); f_seen.add(tid)
        if filtered:
            uniq_candidates = filtered

    # Jazz-intent filter: keep artists with jazz-family tags; exclude latin/urbano roots unless also jazz
    if is_jazzy_query:
        jazzy = []
        j_seen = set()
        for t in uniq_candidates:
            arts = (t.get("artists") or [])
            aid_list = [a.get("id") for a in arts if a.get("id")]
            if not aid_list:
                continue
            artist_has_jazz = any(_has_any_genre(aid, JAZZ_FAMILY_KEYWORDS) for aid in aid_list)
            if not artist_has_jazz:
                continue
            artist_is_excluded = (
                any(_has_any_root(aid, JAZZ_EXCLUDE_ROOTS) for aid in aid_list)
                and not artist_has_jazz
            )
            if artist_is_excluded:
                continue
            tid = t.get("id")
            if tid and tid not in j_seen:
                jazzy.append(t); j_seen.add(tid)
        if jazzy:
            uniq_candidates = jazzy

    # --- Popularity floor to keep out the long tail ---
    if min_popularity > 0:
        uniq_candidates = [t for t in uniq_candidates if int(t.get("popularity") or 0) >= min_popularity]

    # Artist diversification: 1 or 2 tracks per primary artist (pre-rerank)
    cap_per_artist = 1 if (unique_artists or is_jazzy_query) else 2
    artist_count: dict[str, int] = {}
    diversified: list[dict] = []
    for t in uniq_candidates:
        arts = t.get("artists") or []
        primary = (arts[0].get("name") if arts else "Unknown").lower()
        if artist_count.get(primary, 0) < cap_per_artist:
            diversified.append(t)
            artist_count[primary] = artist_count.get(primary, 0) + 1
    uniq_candidates = diversified

    if not uniq_candidates:
        return []

    # Phase 1: SBERT w/o audio features (title boosts if available)
    model = get_sbert_model(sbert_model_name)
    from collections import Counter
    title_boosts: dict[str, float] = {}
    freq_counts = Counter()

    if model is not None and title_map:
        q_emb = model.encode([text], normalize_embeddings=True)
        all_titles = sorted({tt for titles in title_map.values() for tt in titles if tt})
        if all_titles:
            t_embs = model.encode(all_titles, normalize_embeddings=True, batch_size=64)
            sims = util.cos_sim(q_emb, t_embs).cpu().numpy()[0]
            sims = np.maximum(sims, 0.0)
            if sims.max() > 0:
                sims = sims / sims.max()
            title_to_score = {title: float(score) for title, score in zip(all_titles, sims)}
            for tid, titles in title_map.items():
                title_boosts[tid] = sum(title_to_score.get(tt, 0.0) for tt in titles)
                freq_counts[tid] = len(titles)

    shortlist_size = max(SHORTLIST_AFTER_PASS1, limit * 3)
    phase1 = sbert_rerank(
        text,
        uniq_candidates,
        {},                 # NO audio features yet
        genres,
        moods,
        limit=shortlist_size,
        model_name=sbert_model_name,
        freq_counts=freq_counts,
        title_boosts=title_boosts,
        anchor_ratio=min(0.35, anchor_ratio),                 # gentler mixing in phase 1
        anchor_pop_threshold=anchor_pop_threshold,
    )

    # Phase 2: fetch audio features only for shortlist, then final rerank
    feat_map: Dict[str, dict] = {}
    if audio_features_available() and phase1:
        ids = [t["id"] for t in phase1[:shortlist_size]]
        feat_map = get_audio_features(ids)

    final = sbert_rerank(
        text,
        phase1,
        feat_map,
        genres,
        moods,
        limit=max(limit * 2, 30),
        model_name=sbert_model_name,
        freq_counts=freq_counts,
        title_boosts=title_boosts,
        anchor_ratio=anchor_ratio,
        anchor_pop_threshold=anchor_pop_threshold,
    )

    # ---- Post-rerank de-dup & optional one-per-artist cap ----
    def _norm_title(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"\s*\(.*?(remaster|edit|version|mix|mono|stereo|live).*?\)\s*", " ", s)
        s = re.sub(r"\s*-\s*(remaster|edit|version|mix|mono|stereo|live).*?$", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def _primary_artist(t: dict) -> str:
        arts = t.get("artists") or []
        return (arts[0].get("name") if arts else "Unknown").lower()

    # 1) de-dupe (artist,title)
    seen_pair = set()
    deduped = []
    for t in final[: max(limit * 2, 60)]:  # headroom before capping
        key = (_primary_artist(t), _norm_title(t.get("name")))
        if key in seen_pair:
            continue
        seen_pair.add(key)
        deduped.append(t)

    # 2) optional one-per-artist cap AFTER rerank
    if unique_artists:
        seen_art = set()
        uniq_post = []
        for t in deduped:
            pa = _primary_artist(t)
            if pa in seen_art:
                continue
            seen_art.add(pa)
            uniq_post.append(t)
        deduped = uniq_post

    final = deduped

    return format_tracks(final[:limit])

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
    parser.add_argument("--unique-artists", action="store_true", help="Limit to at most one track per primary artist (pre- and post-rerank)")
    parser.add_argument("--one-per-artist", action="store_true", help="Alias of --unique-artists; enforce one per artist after rerank too")
    parser.add_argument("--sbert-model", type=str, default=None, help="sentence-transformers model name (overrides env)")

    # Popularity-mixing knobs
    parser.add_argument("--anchor-ratio", type=float, default=ANCHOR_RATIO_DEFAULT,
                        help="Target fraction of well-known (anchor) tracks in results (0..1)")
    parser.add_argument("--anchor-pop-threshold", type=int, default=ANCHOR_POP_THRESHOLD,
                        help="Track popularity threshold to consider a song an anchor")

    # Popularity floor
    parser.add_argument("--min-popularity", type=int, default=0,
                        help="Drop any track below this Spotify popularity (0..100) before reranking")

    # Cache knobs
    default_cache_ttl = REQUEST_CACHE_TTL  # avoid making REQUEST_CACHE_TTL local
    parser.add_argument("--cache-ttl", type=int, default=default_cache_ttl, help="Override GET cache TTL (seconds)")
    parser.add_argument("--no-cache", action="store_true", help="Disable GET cache")

    args = parser.parse_args()

    # Apply cache overrides
    if args.no_cache:
        # monkey-patch cache funcs to no-op
        def _cache_get_noop(url, params): return None
        def _cache_set_noop(url, params, data): return None
        globals()['_cache_get'] = _cache_get_noop
        globals()['_cache_set'] = _cache_set_noop
    else:
        # safely update the module-level TTL
        try:
            ttl = int(args.cache_ttl)
            if ttl > 0:
                globals()["REQUEST_CACHE_TTL"] = ttl
        except Exception:
            pass  # keep existing TTL on bad input

    model_name = args.sbert_model or SBERT_MODEL_NAME_ENV_DEFAULT
    query = " ".join(args.text).strip() or "happy pop"

    tracks = recommend_from_text(
        query,
        limit=args.limit,
        strict_genre=args.strict_genre,
        sbert_model_name=model_name,
        unique_artists=args.unique_artists or args.one_per_artist,
        anchor_ratio=args.anchor_ratio,
        anchor_pop_threshold=args.anchor_pop_threshold,
        min_popularity=args.min_popularity,
    )

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
