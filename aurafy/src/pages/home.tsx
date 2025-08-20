import React, { useMemo, useState } from "react";

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
      const res = await fetch(`/api/recommend?${params.toString()}`);
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
      className="min-h-screen p-6 max-w-4xl mx-auto text-left text-white"
      style={{
        background: "center-gradient(135deg, #db70cbff 0%, #d903c4ff 55%, #f8acf0ff 100%)",
        backgroundAttachment: "fixed",
        backgroundSize: "cover",
        backdropFilter: "blur(3px)",
        boxShadow: "inset 0 0 100px rgba(153, 11, 11, 0.93)",
      }}
    >
      <h1
        className="text-4xl font-extrabold mb-2"
        style={{ textShadow: "1px 1px 4px rgba(0,0,0,0.4)" }}
      >
        Aurafy
      </h1>
      <p className="text-sm mb-4" style={{ textShadow: "0 0 8px rgba(0,0,0,0.15)" }}>
        Type a vibe and tune the sliders. We’ll fetch and rank tracks from your backend.
      </p>

      {/* Search row */}
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Try: airport jams, jazzy cafe"
          className="border p-2 rounded w-full text-black"
          onKeyDown={(e) => e.key === "Enter" && searchSongs()}
        />
        <button
          onClick={searchSongs}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
          disabled={loading || !query.trim()}
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
          backgroundColor: "rgba(255, 255, 255, 0.1)",
          backdropFilter: "blur(8px)",
          border: "1px solid rgba(255, 255, 255, 0.2)",
          boxShadow: "0 0 15px rgba(255, 182, 193, 0.4)",
        }}
      >
        {/* Result Limit Slider */}
        <div className="flex-1 min-w-[250px]" style={{ marginLeft: "auto" }}>
          <div className="flex items-center justify-between mb-1">
            <label className="font-medium">Result limit</label>
            <span className="text-sm">{limit}</span>
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
          <div className="flex items-center justify-between mb-1">
            <label className="font-medium">Min track popularity</label>
            <span className="text-sm">{minPopularity}</span>
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
          Crunching embeddings & fetching tunes…
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
              style={{ marginLeft: "auto", fontSize: "25px", color: "white", textShadow: "0 0 5px rgba(0,0,0,0.5)" }}
            >
              {t.name}{" "}
              <span className="font-normal text-gray-300" style={{ color: "#ffc1cc" }}>
                by {t.artist}
              </span>
            </div>

            {/* link on far right */}
            <a
              href={`https://open.spotify.com/track/${t.id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="whitespace-nowrap text-sm text-pink-300 hover:underline"
              style={{ marginLeft: "auto", fontSize: "25px" }}
            >
              Open ↗
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
