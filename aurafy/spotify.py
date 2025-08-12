import time, os, base64
import httpx
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

_token_cache = {"access_token": None, "expires_at": 0}

async def get_app_token() -> str:
    # reuse until 60s before expiry
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["access_token"]

    auth = httpx.Auth()
    basic = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
        _token_cache["access_token"] = data["access_token"]
        _token_cache["expires_at"] = time.time() + int(data["expires_in"])
        return _token_cache["access_token"]

async def spotify_get(url: str, params: dict | None = None):
    token = await get_app_token()
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.json()
