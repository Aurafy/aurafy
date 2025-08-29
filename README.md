# aurafy

Aurafy is a mood-to-music recommender that blends semantic search with audio features to deliver playlists that actually fit the vibe. Instead of relying on rigid keyword search, Aurafy lets you type prompts like “airport jams” or “european rave” and instantly get a curated set of tracks.

Features:
- Semantic reranking with SBERT: maps mood prompts to natural language embeddings.
- Spotify API integration: gathers 1,000+ candidate tracks per query.
- Audio feature filtering: boosts alignment on energy, tempo, and danceability.
- Efficient pipeline: optimized batching/caching cuts API calls by 60%.
- Configurable CLI: JSON-based settings for flexible and reproducible queries.
- Low-latency: sub-second reranked results.
