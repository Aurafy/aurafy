import { useState } from "react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [tracks, setTracks] = useState<any[]>([]);

  const searchSongs = async () => {
    const res = await fetch(`/api/recommend?text=${encodeURIComponent(query)}&limit=5`);
    const data = await res.json();
    setTracks(data);
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Aurafy</h1>
      
      <div className="flex gap-2 mb-6">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type your mood..."
          className="border p-2 rounded w-full"
        />
        <button
          onClick={searchSongs}
          className="bg-blue-500 text-white px-4 py-2 rounded"
        >
          Search
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {tracks.map((track) => (
          <div key={track.id} className="border rounded-lg p-4 shadow">
            <img src={track.albumArt} alt={track.album} className="w-full h-40 object-cover rounded" />
            <h2 className="font-semibold mt-2">{track.name}</h2>
            <p className="text-sm text-gray-600">{track.artist}</p>
            {track.previewUrl && (
              <audio controls src={track.previewUrl} className="mt-2 w-full" />
            )}
            <a
              href={`https://open.spotify.com/track/${track.id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 text-sm mt-2 block"
            >
              Open in Spotify
            </a>
          </div>
        ))}
      </div>
    </div>
  );
}
