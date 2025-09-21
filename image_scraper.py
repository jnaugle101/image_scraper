

#!/usr/bin/env python3
import asyncio, hashlib, logging, os, re, sys
from argparse import ArgumentParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import aiohttp
from bs4 import BeautifulSoup
try:
    import PIL.Image as PILImage  # optional verify
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

DEFAULT_UA = "ImageScraper/1.0 (+https://example.com; for personal use)"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}
IMG_URL_IN_STYLE = re.compile(r"url\((['\"]?)([^)'\"]+)\1\)")
SRCSET_SPLIT = re.compile(r"\s*,\s*")
ROBOTS_CACHE = {}  # host -> robots text (parsed with simple rules)
TIMEOUT = aiohttp.ClientTimeout(total=25)

def norm_url(u: str) -> str:
    parts = urlsplit(u)
    scheme = parts.scheme or "http"
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    return f"{scheme}://{netloc}{path}" + (f"?{parts.query}" if parts.query else "")

def host_of(u: str) -> str:
    return urlsplit(u).netloc.lower()

def ext_from_url(u: str) -> str:
    path = urlsplit(u).path
    _, dot, ext = path.rpartition(".")
    return f".{ext.lower()}" if dot else ""

def ext_from_mime(mime: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }.get(mime.lower(), "")

async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> bytes | None:
    try:
        async with session.get(url, timeout=TIMEOUT) as r:
            if r.status != 200:
                return None
            return await r.read()
    except Exception:
        return None

async def fetch_text(session: aiohttp.ClientSession, url: str) -> str | None:
    try:
        async with session.get(url, timeout=TIMEOUT) as r:
            if r.status != 200:
                return None
            ctype = r.headers.get("content-type", "")
            if "text/html" not in ctype:
                return None
            return await r.text(errors="ignore")
    except Exception:
        return None

def parse_images_and_links(html: str, base_url: str) -> tuple[list[str], list[str]]:
    soup = BeautifulSoup(html, "lxml")
    imgs: list[str] = []
    links: list[str] = []

    # <img src> and srcset
    for img in soup.find_all("img"):
        if src := img.get("src"):
            imgs.append(urljoin(base_url, src))
        if srcset := img.get("srcset"):
            for part in SRCSET_SPLIT.split(srcset.strip()):
                cand = part.split()[0]
                if cand:
                    imgs.append(urljoin(base_url, cand))

    # <source srcset> inside <picture>
    for source in soup.find_all("source"):
        if srcset := source.get("srcset"):
            for part in SRCSET_SPLIT.split(srcset.strip()):
                cand = part.split()[0]
                if cand:
                    imgs.append(urljoin(base_url, cand))

    # very basic inline CSS background-image
    for tag in soup.find_all(style=True):
        style = tag["style"]
        for m in IMG_URL_IN_STYLE.finditer(style):
            cand = m.group(2)
            if cand and not cand.startswith("data:"):
                imgs.append(urljoin(base_url, cand))

    # crawlable links
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        links.append(urljoin(base_url, href))

    # unique & keep http(s)
    imgs = [u for u in dict.fromkeys(imgs) if urlsplit(u).scheme in {"http", "https"}]
    links = [u for u in dict.fromkeys(links) if urlsplit(u).scheme in {"http", "https"}]
    return imgs, links

async def head_content_type(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.head(url, timeout=TIMEOUT, allow_redirects=True) as r:
            return r.headers.get("content-type", "").split(";")[0].strip().lower()
    except Exception:
        return ""

def allowed_image(url: str, content_type: str, allowed_exts: set[str], allowed_mimes: set[str]) -> bool:
    ext = ext_from_url(url)
    if ext and ext in allowed_exts:
        return True
    if content_type and content_type in allowed_mimes:
        return True
    return False

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")

async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    outdir: Path,
    min_bytes: int,
    verify_with_pillow: bool,
    seen_hashes: set[str],
) -> bool:
    # Prefer HEAD to filter by content-type quickly
    ctype = await head_content_type(session, url)
    if not allowed_image(url, ctype, ALLOWED_EXTS, ALLOWED_MIMES):
        # still try if looks like image via extension even w/o HEAD
        ext = ext_from_url(url)
        if not ext or ext not in ALLOWED_EXTS:
            return False

    data = await fetch_bytes(session, url)
    if not data or len(data) < min_bytes:
        return False

    if verify_with_pillow and HAVE_PIL:
        try:
            from io import BytesIO
            with PILImage.open(BytesIO(data)) as im:
                im.verify()
        except Exception:
            return False

    digest = sha256(data)
    if digest in seen_hashes:
        return False
    seen_hashes.add(digest)

    ext = ext_from_mime(ctype) or ext_from_url(url) or ".bin"
    fname = sanitize_filename(digest + ext)
    (outdir / fname).write_bytes(data)
    return True

# --- robots.txt (simple) ---
async def can_fetch(session: aiohttp.ClientSession, user_agent: str, url: str, respect: bool) -> bool:
    if not respect:
        return True
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}"
    robots = await get_robots(session, base)
    # very simple Disallow matcher for "*"
    path = parts.path or "/"
    for rule in robots.get("*", []):
        if path.startswith(rule):
            return False
    return True

async def get_robots(session: aiohttp.ClientSession, base: str) -> dict:
    if base in ROBOTS_CACHE:
        return ROBOTS_CACHE[base]
    try:
        async with session.get(base + "/robots.txt", timeout=TIMEOUT) as r:
            text = await r.text(errors="ignore") if r.status == 200 else ""
    except Exception:
        text = ""
    ROBOTS_CACHE[base] = parse_robots(text)
    return ROBOTS_CACHE[base]

def parse_robots(text: str) -> dict:
    # very small parser: collect Disallow rules for "*"
    current_agents = []
    rules = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent:"):
            agent = line.split(":", 1)[1].strip().lower()
            current_agents = [agent]
            rules.setdefault(agent, [])
        elif line.lower().startswith("disallow:"):
            path = line.split(":", 1)[1].strip() or "/"
            for a in current_agents or ["*"]:
                rules.setdefault(a, []).append(path)
    # only keep "*" for our simple check
    return {"*": rules.get("*", [])}

# --- crawler ---
async def crawl(args):
    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ua = args.user_agent or DEFAULT_UA
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)

    seen_pages: set[str] = set()
    seen_hashes: set[str] = set()
    q: asyncio.Queue = asyncio.Queue()

    # seed queue
    for u in args.urls:
        await q.put((norm_url(u), 0))

    pages_crawled = 0
    images_saved = 0

    async with aiohttp.ClientSession(headers={"User-Agent": ua}, connector=connector) as session:
        async def worker():
            nonlocal pages_crawled, images_saved
            while True:
                try:
                    url, depth = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    return
                if url in seen_pages:
                    q.task_done()
                    continue
                if args.page_limit and pages_crawled >= args.page_limit:
                    q.task_done()
                    return

                # robots
                if not await can_fetch(session, ua, url, args.respect_robots):
                    logging.debug("robots disallow: %s", url)
                    seen_pages.add(url)
                    q.task_done()
                    continue

                async with sem:
                    html = await fetch_text(session, url)
                seen_pages.add(url)
                q.task_done()
                if not html:
                    continue

                pages_crawled += 1
                if pages_crawled % 10 == 0:
                    logging.info("pages: %s, images: %s", pages_crawled, images_saved)

                imgs, links = parse_images_and_links(html, url)

                # schedule image downloads
                dl_tasks = []
                for img in imgs:
                    dl_tasks.append(download_image(session, img, outdir, args.min_bytes, args.verify, seen_hashes))
                if dl_tasks:
                    for done in asyncio.as_completed(dl_tasks):
                        try:
                            if await done:
                                images_saved += 1
                        except Exception:
                            pass

                # schedule more pages
                if depth < args.depth:
                    for link in links:
                        if args.same_domain and host_of(link) != host_of(url):
                            continue
                        if link not in seen_pages:
                            await q.put((link, depth + 1))

        workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
        await asyncio.gather(*workers, return_exceptions=True)

    logging.info("DONE — pages crawled: %s, images saved: %s", pages_crawled, images_saved)

def build_arg_parser():
    p = ArgumentParser(description="Minimal, polite image scraper (async).")
    p.add_argument("urls", nargs="+", help="Start URL(s)")
    p.add_argument("--out", default="./downloads", help="Output folder (default: ./downloads)")
    p.add_argument("--depth", type=int, default=0, help="Crawl depth (0 = just the given pages)")
    p.add_argument("--same-domain", action="store_true", help="Only follow links within the same domain")
    p.add_argument("--page-limit", type=int, default=200, help="Max pages to crawl (total)")
    p.add_argument("--concurrency", type=int, default=8, help="Concurrent requests (per host)")
    p.add_argument("--min-bytes", type=int, default=8_000, help="Skip tiny images (< bytes)")
    p.add_argument("--user-agent", default=None, help="Custom User-Agent header")
    p.add_argument("--respect-robots", action="store_true", help="Respect robots.txt (recommended)")
    p.add_argument("--verify", action="store_true", help="Open bytes with Pillow to verify images")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v or -vv for more logs")
    return p

def main():
    args = build_arg_parser().parse_args()
    level = logging.WARNING - min(args.verbose, 2) * 10
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    try:
        asyncio.run(crawl(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)

if __name__ == "__main__":
    main()
