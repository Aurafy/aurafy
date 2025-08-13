#!/usr/bin/env python3
import os
import time
import base64
import argparse
import json
from typing import Dict, List, Tuple, Optional, Set
import tempfile
import re

import httpx
from dotenv import load_dotenv

# =========================== Setup & Globals ===========================

_SPOTIFY_ID_RE = re.compile(r"^[A-Za-z0-9]{22}$")
_artist_genres_cache: dict[str, list[str]] = {}

def _clean_track_ids(ids):
    """Keep only well-formed 22-char base62 IDs."""
    return [tid for tid in ids if tid and _SPOTIFY_ID_RE.match(tid)]

load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")

DEFAULT_MARKET = "US"
SEARCH_PAGES = 3
YEAR_HINT = "2010-2025"
MAX_CANDIDATES = 400
AUDIO_FEATURES_BATCH = 80
AUDIO_FEATURES_AVAILABLE = None  # tri-state: None/True/False
HTTP_TIMEOUT = 30.0

# Local allowlist for genre tokens
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

# Mood presets → audio-feature ranges (used for Spotify features or local proxies)
PRESETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "happy":      {"valence": (0.7, 1.0), "energy": (0.6, 0.95), "danceability": (0.6, 0.95), "tempo": (105, 160)},
    "sad":        {"valence": (0.0, 0.35), "energy": (0.2, 0.6),  "acousticness": (0.4, 1.0),  "tempo": (60, 110)},
    "chill":      {"valence": (0.3, 0.7), "energy": (0.2, 0.5),  "acousticness": (0.3, 0.9),  "tempo": (60, 105)},
    "hype":       {"valence": (0.5, 1.0), "energy": (0.8, 1.0),  "danceability": (0.7, 1.0),  "tempo": (120, 180)},
    "nostalgic":  {"valence": (0.4, 0.75),"energy": (0.3, 0.65), "acousticness": (0.2, 0.8),  "tempo": (70, 120)},
}

# =========================== Optional local analysis deps ===========================
_LIBROSA_OK = None
def _ensure_librosa():
    global _LIBROSA_OK
    if _LIBROSA_OK is not None:
        return _LIBROSA_OK
    try:
        import librosa  # noqa
        import numpy as np  # noqa
        import soundfile as sf  # noqa
        _LIBROSA_OK = True
    except Exception:
        _LIBROSA_OK = False
    return _LIBROSA_OK

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
            headers={"User-Agent": "aurafy/1.0"}
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

def _get(url: str, params: Optional[dict] = None) -> dict:
    token = get_app_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    r = http_client().get(url, params=params, headers=headers)
    r.raise_for_status()
    return r.json()

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

def merge_feature_ranges(moods: List[str]) -> Dict[str, Tuple[float, float]]:
    if not moods:
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    for m in moods:
        for k, (a, b) in PRESETS[m].items():
            if k not in out:
                out[k] = (a, b)
            else:
                aa, bb = out[k]
                out[k] = (min(aa, a), max(bb, b))
    return out

# =========================== Candidate Gathering ===========================
def search_tracks_by_term(term: str, pages: int = SEARCH_PAGES, market: str = DEFAULT_MARKET) -> List[dict]:
    items: List[dict] = []
    queries = [f'track:"" {term} year:{YEAR_HINT}', f'{term} year:{YEAR_HINT}', term]
    seen_ids = set()
    for q in queries:
        for i in range(pages):
            params = {"q": q, "type": "track", "limit": 50, "offset": i * 50, "market": market}
            data = _get("https://api.spotify.com/v1/search", params=params)
            tracks = data.get("tracks", {}).get("items", [])
            if not tracks: break
            for t in tracks:
                tid = t.get("id")
                if tid and tid not in seen_ids:
                    items.append(t); seen_ids.add(tid)
        if items: break
    return items

def search_tracks_by_genres(genres: List[str], pages_per_genre: int = SEARCH_PAGES) -> List[dict]:
    items: List[dict] = []
    for g in genres[:3]:
        items.extend(search_tracks_by_term(g, pages=pages_per_genre, market=DEFAULT_MARKET))
    return items

def expand_via_artists(genres: List[str], limit_artists: int = 10) -> List[dict]:
    track_items: List[dict] = []
    seen = set()
    for g in genres[:3]:
        data = _get("https://api.spotify.com/v1/search", {"q": g, "type": "artist", "limit": limit_artists})
        artists = data.get("artists", {}).get("items", [])
        for a in artists:
            aid = a.get("id")
            if not aid: continue
            tops = _get(f"https://api.spotify.com/v1/artists/{aid}/top-tracks", {"market": DEFAULT_MARKET}).get("tracks", [])
            for t in tops:
                tid = t.get("id")
                if tid and tid not in seen:
                    track_items.append(t); seen.add(tid)
    return track_items

# =========================== Audio Features ===========================
def audio_features_available() -> bool:
    global AUDIO_FEATURES_AVAILABLE
    if AUDIO_FEATURES_AVAILABLE is not None:
        return AUDIO_FEATURES_AVAILABLE
    tid = "0VjIjW4GlUZAMYd2vXMi3b"
    r = http_client().get(
        f"https://api.spotify.com/v1/audio-features/{tid}",
        headers={"Authorization": f"Bearer {get_app_token()}", "Accept":"application/json"}
    )
    AUDIO_FEATURES_AVAILABLE = (r.status_code == 200)
    print(f"[debug] audio_features_available={AUDIO_FEATURES_AVAILABLE}")
    return AUDIO_FEATURES_AVAILABLE

def _get_audio_features_batch(ids_chunk: List[str]) -> Dict[str, dict]:
    feats: Dict[str, dict] = {}
    if not ids_chunk: return feats
    r = http_client().get(
        "https://api.spotify.com/v1/audio-features",
        params={"ids": ",".join(ids_chunk)},
        headers={"Authorization": f"Bearer {get_app_token()}", "Accept":"application/json"},
    )
    if r.status_code == 200:
        data = r.json()
        for f in (data.get("audio_features") or []):
            if f and f.get("id"): feats[f["id"]] = f
    return feats

def _get_audio_features_single(tid: str) -> Optional[dict]:
    r = http_client().get(
        f"https://api.spotify.com/v1/audio-features/{tid}",
        headers={"Authorization": f"Bearer {get_app_token()}", "Accept":"application/json"},
    )
    if r.status_code == 200:
        data = r.json()
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
    failed = 0
    for tid in list(remaining):
        f = _get_audio_features_single(tid)
        if f: feats[tid] = f
        else: failed += 1
    print(f"[debug] audio_features_ok={len(feats)} fallback_failed={failed}")
    return feats

def matches(features: dict, constraints: dict) -> bool:
    tol = 1e-6
    for k, (a, b) in constraints.items():
        if k == "tempo":
            v = features.get("tempo")
            if v is None or v + tol < a or v - tol > b: return False
        else:
            v = features.get(k)
            if v is None: return False
            aa = max(0.0, a); bb = min(1.0, b)
            if v + tol < aa or v - tol > bb: return False
    return True

# =========================== Local Preview Analysis ===========================
def _download_preview(url: str, path: str) -> bool:
    try:
        with httpx.stream("GET", url, timeout=HTTP_TIMEOUT) as r:
            if r.status_code != 200: return False
            with open(path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return True
    except Exception:
        return False

def analyze_preview_file(path: str) -> Optional[dict]:
    if not _ensure_librosa():
        return None
    import numpy as np
    import librosa
    try:
        y, sr = librosa.load(path, sr=22050, mono=True)
        if y.size == 0: return None
        yt, _ = librosa.effects.trim(y, top_db=30)
        if yt.size < sr * 2: yt = y

        tempo, _ = librosa.beat.beat_track(y=yt, sr=sr)
        tempo = float(tempo)

        rms = librosa.feature.rms(y=yt).mean()
        energy = float(np.clip((rms - 0.02) / 0.3, 0.0, 1.0))

        sc = librosa.feature.spectral_centroid(y=yt, sr=sr).mean()
        sc_norm = float(np.clip((sc - 1500) / 2500, 0.0, 1.0))

        zcr = librosa.feature.zero_crossing_rate(yt).mean()

        onset_env = librosa.onset.onset_strength(y=yt, sr=sr)
        beat_strength = float(np.clip(np.median(onset_env) / (np.max(onset_env) + 1e-6), 0.0, 1.0))
        danceability = float(np.clip(beat_strength * (0.5 + 0.5 * (80 <= tempo <= 160)), 0.0, 1.0))

        acousticness = float(np.clip(1.0 - (0.6 * sc_norm + 0.4 * (zcr * 5)), 0.0, 1.0))

        try:
            chroma = librosa.feature.chroma_cqt(y=yt, sr=sr).mean(axis=1)
            major_bias = float(np.clip((chroma.max() - chroma.min()) / (chroma.mean() + 1e-6), 0.0, 1.0))
        except Exception:
            major_bias = 0.5
        valence = float(np.clip(0.6 * sc_norm + 0.4 * major_bias, 0.0, 1.0))

        return {"tempo": tempo, "energy": energy, "danceability": danceability,
                "acousticness": acousticness, "valence": valence}
    except Exception:
        return None

def get_local_features_for_candidates(tracks: List[dict], max_analyze: int = 250) -> Dict[str, dict]:
    feats: Dict[str, dict] = {}
    count = 0
    with tempfile.TemporaryDirectory() as td:
        for t in tracks:
            if count >= max_analyze: break
            tid = t.get("id"); url = t.get("preview_url") or t.get("previewUrl")
            if not tid or not url: continue
            path = os.path.join(td, f"{tid}.mp3")
            if not _download_preview(url, path): continue
            f = analyze_preview_file(path)
            if f:
                feats[tid] = f
                count += 1
    print(f"[debug] local_features_ok={len(feats)} (from previews)")
    return feats

# =========================== Heuristic & Genre Helpers ===========================
def get_artist_genres(aid: str) -> list[str]:
    if not aid: return []
    if aid in _artist_genres_cache: return _artist_genres_cache[aid]
    try:
        data = _get(f"https://api.spotify.com/v1/artists/{aid}")
        gens = data.get("genres") or []
    except Exception:
        gens = []
    _artist_genres_cache[aid] = gens
    return gens

def artist_matches_any_genre(artist_id: str, targets: list[str]) -> bool:
    gens = [g.lower() for g in get_artist_genres(artist_id)]
    tgt = [t.lower() for t in targets]
    return any(any(t in g for t in tgt) for g in gens)

def heuristic_rank_tracks(tracks: list[dict], tokens: list[str], limit: int) -> list[dict]:
    W_POP = 1.0; W_HAPPY = 6.0; W_POPGEN = 4.0
    happy_words = {"happy","feel good","feel-good","sun","sunny","summer","smile","bright","dance","party","fun","love","feel"}
    def score(t: dict) -> float:
        s = float(t.get("popularity") or 0)
        name = (t.get("name") or "").lower()
        if any(w in name for w in happy_words): s += W_HAPPY
        gens = []
        for a in (t.get("artists") or []): gens.extend(get_artist_genres(a.get("id")))
        gl = [g.lower() for g in gens]
        if any(("pop" in g) or ("dance" in g) or ("electro" in g) for g in gl): s += W_POPGEN
        return s
    return sorted(tracks, key=score, reverse=True)[:limit]

# =========================== Utils ===========================
def _relax(constraints: dict, widen_pct: float = 0.2, widen_bpm: float = 20) -> dict:
    out = {}
    for k, (a, b) in constraints.items():
        if k == "tempo": out[k] = (max(0, a - widen_bpm), b + widen_bpm)
        else: out[k] = (max(0.0, a * (1 - widen_pct)), min(1.0, b * (1 + widen_pct)))
    return out

def _as_track(obj: dict | None) -> dict | None:
    if not obj or not isinstance(obj, dict): return None
    if obj.get("type") != "track": return None
    if not obj.get("id"): return None
    return obj

def collect_from_playlists(term: str, max_playlists: int = 3, max_tracks_per: int = 120) -> list[dict]:
    data = _get("https://api.spotify.com/v1/search", {"q": term, "type": "playlist", "limit": max_playlists})
    pl_items = (data.get("playlists") or {}).get("items") or []
    out: list[dict] = []; seen: set[str] = set()
    for pl in pl_items:
        if not pl or not isinstance(pl, dict): continue
        pid = pl.get("id")
        if not pid: continue
        pulled = 0; offset = 0
        while pulled < max_tracks_per:
            resp = _get(f"https://api.spotify.com/v1/playlists/{pid}/tracks",
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
    return out

# =========================== Preview-Aware Augmentation ===========================
def _combine_terms(moods: List[str], genres: List[str], allow_covers: bool) -> List[str]:
    """
    Build combined search terms to bias toward preview-available tracks.
    If allow_covers=True, include cover/lofi/acoustic/instrumental keywords.
    """
    base = []
    if genres: base.extend(genres)
    if moods: base.extend(moods)
    combos = set()
    toks = base[:]
    # singles
    for t in toks: combos.add(t)
    # pairs like "sad disney"
    for m in moods:
        for g in genres:
            combos.add(f"{m} {g}")
            combos.add(f"{g} {m}")
    # playlist-y terms
    for g in genres:
        combos.add(f"{g} playlist"); combos.add(f"{g} hits")
    if allow_covers:
        modifiers = ["cover", "covers", "piano", "acoustic", "instrumental", "lofi", "lo-fi"]
        for g in genres or ["pop"]:
            for mod in modifiers:
                combos.add(f"{g} {mod}")
                if moods:
                    for m in moods:
                        combos.add(f"{m} {g} {mod}")
                        combos.add(f"{g} {mod} {m}")
    return list(combos)

def augment_with_preview_tracks(moods: List[str], genres: List[str], allow_covers: bool, target_min: int = 100) -> List[dict]:
    """
    Try extra searches and playlists using combined terms. Return tracks that have preview_url.
    Aggressively pulls cover/lofi/acoustic variants when allow_covers=True.
    """
    terms = _combine_terms(moods, genres, allow_covers)
    out: List[dict] = []
    seen = set()
    # Search tracks
    for term in terms:
        for t in search_tracks_by_term(term, pages=2, market=DEFAULT_MARKET):
            if (t.get("preview_url") or t.get("previewUrl")) and t.get("id") and t["id"] not in seen:
                out.append(t); seen.add(t["id"])
                if len(out) >= target_min: return out
    # Playlists
    for term in terms[:6]:
        for t in collect_from_playlists(term, max_playlists=2, max_tracks_per=150):
            if (t.get("preview_url") or t.get("previewUrl")) and t.get("id") and t["id"] not in seen:
                out.append(t); seen.add(t["id"])
                if len(out) >= target_min: return out
    return out

# =========================== Core Recommend ===========================
def recommend_from_text(text: str, limit: int, *, allow_heuristic: bool, strict_genre: bool, local_audio: bool, allow_covers: bool) -> List[dict]:
    moods, genres = parse_query(text)
    if not genres:
        for g in ["pop","indie-pop","dance","rock","edm","hip-hop"]:
            if g in ALLOWED_GENRES: genres = [g]; break
        if not genres: genres = ["pop"]
    constraints = merge_feature_ranges(moods) if moods else PRESETS["chill"]

    # Candidate pool
    candidates = search_tracks_by_genres(genres, pages_per_genre=SEARCH_PAGES)
    for g in genres[:2]:
        candidates.extend(collect_from_playlists(g, max_playlists=2, max_tracks_per=150))
    if len(candidates) < 150:
        candidates.extend(expand_via_artists(genres))

    # Dedup + cap
    seen = set(); uniq_candidates = []
    for t in candidates:
        tid = t.get("id")
        if tid and tid not in seen:
            uniq_candidates.append(t); seen.add(tid)
    if len(uniq_candidates) > MAX_CANDIDATES:
        uniq_candidates = uniq_candidates[:MAX_CANDIDATES]

    # Optional strict genre filter via artist genres
    if strict_genre and genres:
        genre_tokens = genres[:]
        filtered_by_genre = []
        seen_ids = set()
        for t in uniq_candidates:
            arts = t.get("artists") or []
            if any(artist_matches_any_genre(a.get("id"), genre_tokens) for a in arts):
                tid = t.get("id")
                if tid and tid not in seen_ids:
                    filtered_by_genre.append(t); seen_ids.add(tid)
        if filtered_by_genre:
            uniq_candidates = filtered_by_genre

    print(f"[debug] candidates={len(candidates)}, unique={len(uniq_candidates)}")
    if not uniq_candidates: return []

    # Spotify features path (if available and not using local)
    af_ok = audio_features_available()
    if af_ok and not local_audio:
        ids = [t["id"] for t in uniq_candidates]
        feat_map = get_audio_features(ids)
        print(f"[debug] with_features={len(feat_map)}")

        def filter_with(cs: dict) -> list[dict]:
            out = []
            for t in uniq_candidates:
                af = feat_map.get(t["id"])
                if af and matches(af, cs):
                    out.append(t)
                    if len(out) >= 3 * limit: break
            return out

        filtered = filter_with(constraints)
        if len(filtered) < limit:
            filtered = filter_with(_relax(constraints, widen_pct=0.3, widen_bpm=30))
        if len(filtered) < limit:
            relaxed = dict(constraints)
            for drop in ("acousticness","tempo","danceability"):
                if drop in relaxed:
                    relaxed.pop(drop)
                    filtered = filter_with(_relax(relaxed, widen_pct=0.35, widen_bpm=35))
                    if len(filtered) >= limit: break
        if len(filtered) < limit:
            filtered = sorted((t for t in uniq_candidates if t.get("popularity") is not None),
                              key=lambda x: x["popularity"], reverse=True)[:limit*2]
        chosen = filtered

    else:
        # Local preview analysis or heuristic
        preview_first = [t for t in uniq_candidates if t.get("preview_url") or t.get("previewUrl")]

        # If previews are scarce, aggressively augment with covers/instrumentals/lofi (opt-in)
        if len(preview_first) < 60:
            extra = augment_with_preview_tracks(moods, genres, allow_covers=allow_covers, target_min=120)
            seen_ids = {t.get("id") for t in preview_first}
            for t in extra:
                tid = t.get("id")
                if tid and tid not in seen_ids:
                    preview_first.append(t); seen_ids.add(tid)

        print(f"[debug] preview_candidates={len(preview_first)}")

        if local_audio:
            if not _ensure_librosa():
                raise SystemExit("Local preview analysis requires: pip install librosa soundfile numpy scipy")

            feat_map = get_local_features_for_candidates(preview_first, max_analyze=300)

            if not feat_map:
                if allow_heuristic:
                    chosen = heuristic_rank_tracks(preview_first or uniq_candidates, genres + moods, limit*2)
                else:
                    raise SystemExit(
                        "No 30s previews available to analyze locally for this query. "
                        "Try without --local-audio or pass --allow-heuristic (or add --allow-covers to include preview-heavy covers)."
                    )
            else:
                def filter_with_local(cs: dict) -> list[dict]:
                    out = []
                    for t in preview_first:
                        af = feat_map.get(t.get("id"))
                        if af and matches(af, cs):
                            out.append(t)
                            if len(out) >= 3 * limit: break
                    return out

                filtered = filter_with_local(constraints)
                if len(filtered) < limit:
                    filtered = filter_with_local(_relax(constraints, widen_pct=0.35, widen_bpm=35))
                if len(filtered) < limit:
                    # choose closest by distance to constraints
                    def dist(t):
                        af = feat_map.get(t.get("id"))
                        if not af: return 1e9
                        d = 0.0
                        for k,(a,b) in constraints.items():
                            v = af.get(k)
                            if v is None: d += 1.0
                            else:
                                if k=="tempo":
                                    if v < a: d += (a-v)/max(a,1)
                                    elif v > b: d += (v-b)/max(b,1)
                                else:
                                    if v < a: d += (a-v)
                                    elif v > b: d += (v-b)
                        return d
                    candidates_with_feats = [t for t in preview_first if feat_map.get(t.get("id"))]
                    filtered = sorted(candidates_with_feats, key=dist)[:limit*2]
                chosen = filtered

        else:
            if not allow_heuristic:
                raise SystemExit(
                    "Audio features endpoint is not reachable, and --local-audio not enabled.\n"
                    "Use --local-audio to analyze previews locally, or run with --allow-heuristic.\n"
                    "Tip: add --allow-covers to include preview-heavy covers for local analysis."
                )
            chosen = heuristic_rank_tracks(preview_first or uniq_candidates, genres + moods, limit*2)

    # Format output
    out: List[dict] = []
    for t in chosen[:max(limit, 0)]:
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

# =========================== CLI ===========================
def main():
    parser = argparse.ArgumentParser(description="Mood2Music: parameter-filtered recs via Spotify or local preview analysis")
    parser.add_argument("text", nargs="*", help="e.g. 'sad disney' or 'chill r-n-b'")
    parser.add_argument("--limit", type=int, default=12, help="number of tracks")
    parser.add_argument("--allow-heuristic", action="store_true", help="allow popularity/genre heuristics if no features/previews")
    parser.add_argument("--strict-genre", action="store_true", help="filter to artists whose genres contain the input genre tokens")
    parser.add_argument("--local-audio", action="store_true", help="compute features from 30s previews (librosa) when Spotify features unavailable")
    parser.add_argument("--allow-covers", action="store_true", help="include covers/instrumentals/lofi/acoustic to increase preview availability for local analysis")
    args = parser.parse_args()

    query = " ".join(args.text).strip() or "happy pop"
    _ = audio_features_available()  # probe once

    tracks = recommend_from_text(
        query,
        limit=args.limit,
        allow_heuristic=args.allow_heuristic,
        strict_genre=args.strict_genre,
        local_audio=args.local_audio,
        allow_covers=args.allow_covers
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
