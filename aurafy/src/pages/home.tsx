import React, { useEffect, useMemo, useRef, useState } from "react";

// ------------------------------------------------------------
// Minimal types to match main.py's format_tracks output
// ------------------------------------------------------------

type Track = {
  id: string;
  name: string;
  artist: string;
  album?: string;
  albumArt?: string;
  previewUrl?: string;
  uri?: string;
  popularity?: number;
};

type RecommendParams = {
  text: string;
  limit?: number;
  strict_genre?: boolean;
  unique_artists?: boolean;
  anchor_ratio?: number; // 0..1
  anchor_pop_threshold?: number; // 0..100
  min_popularity?: number; // 0..100
};

// ------------------------------------------------------------
// Helper: debounce
// ------------------------------------------------------------
function useDebounced<T>(value: T, ms = 350) {
  const [v, setV] = useState(value);
  useEffect(() => { const id = setTimeout(() => setV(value), ms); return () => clearTimeout(id); }, [value, ms]);
  return v;
}

// ------------------------------------------------------------
// Thin client that expects a backend route that shells into main.py
//   GET /api/recommend?text=...&limit=...&strict_genre=true/false&unique_artists=true/false&anchor_ratio=...&anchor_pop_threshold=...&min_popularity=...
// and responds with JSON: Track[]
// ------------------------------------------------------------
async function fetchRecommendations(params: RecommendParams, signal?: AbortSignal): Promise<Track[]> {
  const qp = new URLSearchParams();
  qp.set("text", params.text || "");
  qp.set("limit", String(params.limit ?? 12));
  if (params.strict_genre != null) qp.set("strict_genre", String(params.strict_genre));
  if (params.unique_artists != null) qp.set("unique_artists", String(params.unique_artists));
  if (params.anchor_ratio != null) qp.set("anchor_ratio", String(params.anchor_ratio));
  if (params.anchor_pop_threshold != null) qp.set("anchor_pop_threshold", String(params.anchor_pop_threshold));
  if (params.min_popularity != null) qp.set("min_popularity", String(params.min_popularity));

  const res = await fetch(`/api/recommend?${qp.toString()}`, { signal });
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return (await res.json()) as Track[];
}

// ------------------------------------------------------------
// Quick prompts
// ------------------------------------------------------------
const QUICK: string[] = [
  "jazzy cafe",
  "melancholy nostalgic",
  "energetic bubblegum pop",
  "psychedelic rock trip",
  "ambient focus",
  "r&b slow jam",
  "happy summer indie",
  "rainy late night drive",
];

// ------------------------------------------------------------
// Styles (super barebones, no external UI libs required)
// ------------------------------------------------------------
const styles = {
  page: {
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
    padding: "24px",
    background: "#fafafa",
    minHeight: "100vh",
  } as React.CSSProperties,
  shell: { maxWidth: 1100, margin: "0 auto" } as React.CSSProperties,
  header: { display: "flex", alignItems: "center", gap: 12 } as React.CSSProperties,
  title: { fontSize: 24, fontWeight: 700 } as React.CSSProperties,
  subtitle: { color: "#666", fontSize: 13 } as React.CSSProperties,
  card: { background: "#fff", borderRadius: 12, padding: 16, boxShadow: "0 1px 3px rgba(0,0,0,0.06)", border: "1px solid #eee" } as React.CSSProperties,
  row: { display: "flex", gap: 10, alignItems: "center" } as React.CSSProperties,
  col: { display: "flex", flexDirection: "column", gap: 8 } as React.CSSProperties,
  input: { height: 44, padding: "0 12px", borderRadius: 8, border: "1px solid #ddd", flex: 1 } as React.CSSProperties,
  button: { height: 44, padding: "0 14px", borderRadius: 8, background: "#111", color: "#fff", border: "1px solid #111", cursor: "pointer" } as React.CSSProperties,
  chip: { border: "1px solid #ddd", borderRadius: 999, padding: "6px 10px", fontSize: 12, background: "#fff", cursor: "pointer" } as React.CSSProperties,
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 16 } as React.CSSProperties,
  imgWrap: { position: "relative", width: "100%", paddingTop: "56%", background: "#f0f0f0", borderRadius: 10, overflow: "hidden" } as React.CSSProperties,
  img: { position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" } as React.CSSProperties,
  small: { fontSize: 12, color: "#666" } as React.CSSProperties,
};

// ------------------------------------------------------------
// Component
// ------------------------------------------------------------
export default function Home() {
  const [text, setText] = useState("happy pop");
  const [limit, setLimit] = useState(12);
  const [strict, setStrict] = useState(false);
  const [unique, setUnique] = useState(false);
  const [minPop, setMinPop] = useState(0);
  const [anchorRatio, setAnchorRatio] = useState(0.6);
  const [anchorThresh, setAnchorThresh] = useState(65);

  const [tracks, setTracks] = useState<Track[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const debouncedText = useDebounced(text, 350);
  const canSearch = useMemo(() => debouncedText.trim().length > 0, [debouncedText]);

  async function search() {
    if (!canSearch) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true); setError(null);
    try {
      const data = await fetchRecommendations({
        text: debouncedText,
        limit,
        strict_genre: strict,
        unique_artists: unique,
        min_popularity: minPop,
        anchor_ratio: anchorRatio,
        anchor_pop_threshold: anchorThresh,
      }, ctrl.signal);
      if (!ctrl.signal.aborted) setTracks(data);
    } catch (e: any) {
      if (!ctrl.signal.aborted) setError(e?.message || "Something went wrong");
    } finally {
      if (!ctrl.signal.aborted) setLoading(false);
    }
  }

  useEffect(() => { search(); /* auto-run when knobs/text change */
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedText, limit, strict, unique, minPop, anchorRatio, anchorThresh]);

  return (
    <div style={styles.page}>
      <div style={styles.shell}>
        <header style={styles.header}>
          <div>
            <div style={styles.title}>Aurafy</div>
            <div style={styles.subtitle}>Type a vibe or genres — we'll fetch tracks via main.py</div>
          </div>
        </header>

        {/* Search */}
        <section style={{ marginTop: 16 }}>
          <div style={styles.card}>
            <div style={{ ...styles.row, gap: 8 }}>
              <input
                style={styles.input}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="e.g., jazzy cafe, energetic bubblegum pop, rainy late night drive"
                onKeyDown={(e) => { if (e.key === 'Enter') search(); }}
              />
              <button style={styles.button} onClick={search} disabled={!canSearch || loading}>
                {loading ? "Searching…" : "Search"}
              </button>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 10 }}>
              {QUICK.map(q => (
                <button key={q} style={styles.chip} onClick={() => setText(q)}>{q}</button>
              ))}
            </div>
          </div>
        </section>

        {/* Knobs */}
        <section style={{ marginTop: 12 }}>
          <div style={styles.card}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
              <div style={styles.col}>
                <label>Limit: {limit}</label>
                <input type="range" min={5} max={50} step={1} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
              </div>
              <div style={styles.col}>
                <label>Min Popularity: {minPop}</label>
                <input type="range" min={0} max={90} step={5} value={minPop} onChange={(e) => setMinPop(Number(e.target.value))} />
              </div>
              <div style={styles.col}>
                <label>Anchor Ratio: {anchorRatio.toFixed(2)}</label>
                <input type="range" min={0} max={1} step={0.05} value={anchorRatio} onChange={(e) => setAnchorRatio(Number(e.target.value))} />
              </div>
              <div style={styles.col}>
                <label>Anchor Popularity Threshold: {anchorThresh}</label>
                <input type="range" min={40} max={90} step={5} value={anchorThresh} onChange={(e) => setAnchorThresh(Number(e.target.value))} />
              </div>
              <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <input type="checkbox" checked={strict} onChange={(e) => setStrict(e.target.checked)} /> Strict genre
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <input type="checkbox" checked={unique} onChange={(e) => setUnique(e.target.checked)} /> One per artist
              </label>
            </div>
          </div>
        </section>

        {/* Results */}
        <section style={{ marginTop: 16 }}>
          {error && (
            <div style={{ color: "#b00020", marginBottom: 12, ...styles.card }}>
              {error}
            </div>
          )}

          {!error && loading && (
            <div style={{ ...styles.card }}>Crunching embeddings & fetching tunes…</div>
          )}

          {!loading && tracks && tracks.length === 0 && (
            <div style={{ ...styles.card }}>No results. Try loosening Min Popularity or disabling Strict genre.</div>
          )}

          {!loading && tracks && tracks.length > 0 && (
            <div style={styles.grid}>
              {tracks.map(t => (
                <div key={t.id} style={styles.card}>
                  <div style={{ fontWeight: 600, lineHeight: 1.2 }}>{t.name}</div>
                  <div style={styles.small}>{t.artist}</div>
                  <div style={{ height: 8 }} />
                  <div style={styles.imgWrap}>
                    {t.albumArt ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={t.albumArt} alt={`${t.name} cover`} style={styles.img} />
                    ) : null}
                  </div>
                  <div style={{ height: 8 }} />
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={styles.small}>Popularity {t.popularity ?? 0}</div>
                    {t.uri && (
                      <a href={t.uri} target="_blank" rel="noreferrer" style={{ fontSize: 12 }}>Open in Spotify</a>
                    )}
                  </div>
                  {t.previewUrl && (
                    <audio style={{ marginTop: 8, width: "100%" }} src={t.previewUrl} controls preload="none" />
                  )}
                </div>
              ))}
            </div>
          )}
        </section>

        <footer style={{ marginTop: 20 }}>
          <div style={styles.small}>Backend endpoint assumed at <code>/api/recommend</code>. Wire this to run <code>main.py</code> and return JSON from <code>format_tracks</code>.</div>
        </footer>
      </div>
    </div>
  );
}
