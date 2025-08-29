import { useMemo, useState } from "react";
import { API } from "../config";
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

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Track[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [limit, setLimit] = useState<number>(15);
  const [minPopularity, setMinPopularity] = useState<number>(30);

  const requestLimit = useMemo(() => {
    if (minPopularity <= 0) return limit;
    return Math.min(60, Math.max(limit, Math.ceil(limit * 2)));
  }, [limit, minPopularity]);

  const searchSongs = async () => {
  if (!query.trim()) return;
  setLoading(true);
  setError(null);
  try {
    const params = new URLSearchParams({
      text: query,
      limit: String(requestLimit),
      min_popularity: String(minPopularity),
    });

    // Use the API base (env-driven) instead of a relative /api path
    const res = await fetch(`${API}/api/recommend?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = (await res.json()) as Track[];
    setResults(Array.isArray(data) ? data : []);
  } catch (e: any) {
    setError(e?.message || "Something went wrong");
  } finally {
    setLoading(false);
  }
};

  const filtered = useMemo(
    () => results.filter((t) => (t.popularity ?? 0) >= minPopularity),
    [results, minPopularity]
  );
  const displayed = useMemo(() => filtered.slice(0, limit), [filtered, limit]);

  return (
    <div
      className="p-6 max-w-4xl mx-auto text-left text-white"
      style={{
        fontFamily: "'Baloo 2', cursive",
        backdropFilter: "blur(3px)",
        color: "white"
      }}
    >
      <h1
        className="text-4xl font-extrabold mb-2"
        style = {{ fontSize: "60px", marginBottom: "1rem"}}
      >
        ▶︎•၊၊||၊|။ Aurafy |||||။၊|။•
      </h1>
      <p className="text-sm mb-4"
      style = {{ fontSize: "30px", marginTop: "1rem"}}>
       Welcome to Aurafy! Enter a mood, vibe, or theme to discover music that matches your feelings.
      </p>

      {/* Search row */}
<div style={{ display: "flex", gap: "12px", marginBottom: "24px", alignItems: "center", justifyContent: "center" }}>
  <input
    id = "searchInput"
    type="text"
    value={query}
    onChange={(e) => setQuery(e.target.value)}
    placeholder="airport jams, boardwalk ice cream, rainy late night drive, etc"
    style={{
      width: "50%",
      padding: "12px 16px",
      borderRadius: "500px",
      background: "rgba(255,255,255,0.1)",
      color: "white",
      fontSize: "18px",
      border: "none",
      outline: "none",
      boxShadow: "0 4px 10px rgba(97, 83, 83, 0.2)",
    }}
    onKeyDown={(e) => e.key === "Enter" && searchSongs()}
  />
  <style>
  {`
    #searchInput::placeholder {
      color: white;
      opacity: 0.8;
    }
  `}
</style>
  <button
    onClick={searchSongs}
    disabled={loading || !query.trim()}
    style={{
      padding: "12px 24px",
      borderRadius: "16px",
      background: "rgb(203, 67, 135)", // pink → purple
      color: "white",
      fontWeight: 600,
      fontSize: "18px",
      border: "none",
      cursor: loading || !query.trim() ? "not-allowed" : "pointer",
      boxShadow: "0 6px 14px rgba(0,0,0,0.25)",
      transition: "all 0.2s ease-in-out",
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.transform = "scale(1.05)";
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.transform = "scale(1)";
    }}
  >
    {loading ? "Searching…" : "Search"}
  </button>
</div>


      {/* sliders */}
      <div
        className="p-3 rounded-lg flex items-center gap-4"
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          gap: "1rem",
          marginLeft: "auto",
          marginBottom: "3rem"
        }}
      >
        {/* Result Limit Slider */}
        <div className="flex-1 min-w-[250px]" style={{ marginLeft: "auto", marginRight: "7rem" }}>
          <div className="flex items-center justify-between mb-1"  style = {{fontSize: "25px"}}>
            <label className="font-medium" style={{}}># of Tracks: </label>
            <span className="text-sm" style={{ fontWeight: "bold", color: "rgb(247, 240, 213)"}}>{limit}</span>
          </div>
          <input
            type="range"
            min={5}
            max={50}
            step={1}
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Min popularity slider */}
        <div className="flex-1 min-w-[250px]" style={{ marginRight: "auto" }}>
          <div className="flex items-center justify-between mb-1" style = {{fontSize: "25px"}}>
            <label className="font-medium">Track popularity / 100: </label>
            <span className="text-sm" style={{ fontWeight: "bold", color: "rgb(247, 240, 213)"}}>{minPopularity}</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={minPopularity}
            onChange={(e) => setMinPopularity(Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {error && (
        <div className="mb-4 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}
      {!error && loading && (
        <div className="mb-4 text-sm" style={{ textShadow: "0 0 8px rgba(0,0,0,0.15)" }}>
         Fetching tunes…
        </div>
      )}

      {/* RESULTS LIST */}
      <div className="mt-6 space-y-3 w-fit mx-auto self-center">
        {displayed.map((t) => (
          <div
            key={t.id}
            className="p-3 rounded-lg flex items-center gap-4 hover:shadow-[0_0_25px_rgba(255,182,193,0.4)] transition duration-300"
            style={{
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
              gap: "1rem",
              backgroundColor: "rgba(255, 255, 255, 0.05)",
              backdropFilter: "blur(6px)",
              border: "1px solid rgba(255,255,255,0.1)",
            }}
          >
            {/* album art */}
            <a
              href={`https://open.spotify.com/track/${t.id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="shrink-0"
            >
              <img
                src={t.albumArt || ""}
                alt={t.album || t.name}
                className="block rounded object-cover shadow-[0_0_15px_rgba(255,182,193,0.4)]"
                style={{ width: 100, height: 100 }}
              />
            </a>

            {/* song + artist inline */}
            <div
              className="text-lg font-bold truncate"
              style={{ marginLeft: "auto", fontSize: "25px", color: "white" }}
            >
              {t.name}{" "}
              <span className="font-normal text-gray-300" style={{ color: "#ffffff" }}>
                by {t.artist}
              </span>
            </div>

            {/* link on far right */}
            <a
              href={`https://open.spotify.com/track/${t.id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="whitespace-nowrap text-sm text-pink-300 hover:underline"
              style={{ marginLeft: "auto", marginRight: "1rem", fontSize: "20px", textDecoration: "underline" }}
            >
              Open
            </a>
          </div>
        ))}
      </div>

      {!loading && !error && displayed.length === 0 && (
        <p className="mt-4 text-sm text-gray-200" style={{ textShadow: "0 0 8px rgba(0,0,0,0.15)" }}>
          No results yet. Try a search, lower the popularity filter, or increase the limit.
        </p>
      )}
    </div>
  );
}
