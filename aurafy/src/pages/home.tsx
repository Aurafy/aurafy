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
    <div className="p-6 max-w-4xl mx-auto text-left">
      <h1 className="text-4xl font-extrabold mb-2">Aurafy</h1>
      <p className="text-sm text-gray-600 mb-4">
        Type a vibe and tune the sliders. We’ll fetch and rank tracks from your backend.
      </p>

      {/* Search row */}
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Try: white girl pop, jazzy cafe, late night rave"
          className="border p-2 rounded w-full"
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

      {/* Sliders */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6 rounded-lg border p-4">
        <div>
          <div className="flex items-center justify-between">
            <label className="font-medium">Result limit</label>
            <span className="text-sm text-gray-600">{limit}</span>
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
        <div>
          <div className="flex items-center justify-between">
            <label className="font-medium">Min track popularity</label>
            <span className="text-sm text-gray-600">{minPopularity}</span>
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
        <div className="mb-4 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>
      )}
      {!error && loading && (
        <div className="mb-4 text-sm text-gray-600">Crunching embeddings & fetching tunes…</div>
      )}
      {displayed.length > 0 && (
        <div className="text-xs text-gray-600 mb-3">
          Showing {displayed.length} of {limit} (filtered from {results.length}).
        </div>
      )}

      <div role="list" className="rounded-lg border divide-y">
  {displayed.map((t) => (
    <div
      role="listitem"
      key={t.id}
      className="flex items-center px-3 py-2 hover:bg-gray-50"
    >
      {/* Album Cover */}
      <a
        href={`https://open.spotify.com/track/${t.id}`}
        target="_blank"
        rel="noopener noreferrer"
        className="shrink-0 mr-4"
        title="Open in Spotify"
      >
        <img
          src={t.albumArt || ""}
          alt={t.album || t.name}
          style={{
            width: 64,
            height: 64,
            objectFit: "cover",
            borderRadius: 6,
            display: "block",
          }}
          onError={(e: any) => {
            e.currentTarget.style.visibility = "hidden";
            e.currentTarget.style.width = "0px";
          }}
        />
      </a>

      {/* Track Info & Link */}
      <div className="flex flex-1 items-center justify-between min-w-0">
        <div className="min-w-0 truncate">
          <span className="font-medium truncate">{t.name}</span>
          <span className="text-gray-500"> &nbsp;by&nbsp; </span>
          <span className="text-gray-800 truncate">{t.artist}</span>
        </div>

        <a
          href={`https://open.spotify.com/track/${t.id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 text-blue-600 hover:underline text-sm ml-4"
        >
          Open ↗
        </a>
      </div>
    </div>
  ))}
</div>


      {!loading && !error && displayed.length === 0 && (
        <p className="mt-4 text-sm text-gray-500">
          No results yet. Try a search, lower the popularity filter, or increase the limit.
        </p>
      )}
    </div>
  );
}
