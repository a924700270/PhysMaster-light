import aiohttp
import asyncio
import os, sys
from bs4 import BeautifulSoup

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..', '..', '..', 'api_proxy'))

API_TIMEOUT = 15
DIRECT_TIMEOUT = 60

async def _fetch_via_api(session, url: str):
    """Try fetching through the local tool API proxy."""
    from tool_api import fetch_web_api
    try:
        result = await asyncio.wait_for(
            fetch_web_api(session, url),
            timeout=API_TIMEOUT
        )
        if isinstance(result, (list, tuple)) and len(result) == 2:
            return result[0], result[1]
        if isinstance(result, dict):
            content = result.get("content") or result.get("text") or result.get("html") or ""
            success = result.get("success", result.get("status") == "ok" or bool(content))
            return bool(success), str(content)
        return False, ""
    except Exception as e:
        print(f"API proxy unavailable: {e}")
        return None, None


def _html_to_text(html: str) -> str:
    """Convert raw HTML to readable text using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


MAX_READ_BYTES = 2 * 1024 * 1024  # 2MB max to avoid huge pages timing out


async def _fetch_direct(url: str):
    """Directly fetch and parse HTML content with size limit."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    timeout = aiohttp.ClientTimeout(total=DIRECT_TIMEOUT, sock_read=30)
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=True, ssl=False) as resp:
                if resp.status != 200:
                    return False, f"HTTP {resp.status}"
                # Read in chunks with size limit to avoid timeout on huge pages
                chunks = []
                total = 0
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= MAX_READ_BYTES:
                        break
                raw = b"".join(chunks)
                encoding = resp.charset or "utf-8"
                html = raw.decode(encoding, errors="replace")
                text = _html_to_text(html)
                if not text.strip():
                    return False, "Empty page content"
                return True, text
    except Exception as e:
        return False, f"Direct fetch failed: {type(e).__name__}: {e}"


async def fetch_web_content(url: str):
    """
    Fetch web content with fallback:
    1. Try local API proxy (fast, may have JS rendering)
    2. Fall back to direct aiohttp + BeautifulSoup
    """
    async with aiohttp.ClientSession() as session:
        api_ok, api_text = await _fetch_via_api(session, url)
        if api_ok is not None and api_ok and api_text:
            return True, api_text

    print("Falling back to direct HTTP fetch...")
    return await _fetch_direct(url)


async def main():
    url = "https://proceedings.neurips.cc/paper_files/paper/2022"
    is_ok, html = await fetch_web_content(url)
    if is_ok:
        print(html[:2000])
    else:
        print(f"fetch failed: {html}")

if __name__ == "__main__":
    asyncio.run(main())
