

#!/usr/bin/env python3
import asyncio, logging, re, io, hashlib, zipfile
from urllib.parse import urljoin, urlsplit
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
import streamlit as st

try:
    from PIL import Image as PILImage  # optional verification
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

from bs4 import BeautifulSoup

def make_soup(html: str) -> BeautifulSoup:
    try:
        # Use lxml if Render happened to build it; otherwise fallback
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")

# ---------- Config ----------
DEFAULT_UA = "ImageScraper/1.0 (+for personal use)"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
ALLOWED_MIMES = {"image/jpeg","image/png","image/gif","image/webp","image/bmp"}
TIMEOUT = aiohttp.ClientTimeout(total=25)
SRCSET_SPLIT = re.compile(r"\s*,\s*")
IMG_URL_IN_STYLE = re.compile(r"url\((['\"]?)([^)'\"]+)\1\)")
ROBOTS_CACHE = {}  # host -> {"*":[disallows]}

# ---------- Small utils ----------
def norm_url(u: str) -> str:
    parts = urlsplit(u.strip())
    if not parts.scheme:
        u = "https://" + u
        parts = urlsplit(u)
    path = parts.path or "/"
    return f"{parts.scheme}://{parts.netloc}{path}" + (f"?{parts.query}" if parts.query else "")

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
    }.get((mime or "").split(";")[0].strip().lower(), "")

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ---------- Fetchers ----------
async def fetch_text(session: aiohttp.ClientSession, url: str) -> str | None:
    try:
        async with session.get(url, timeout=TIMEOUT) as r:
            if r.status != 200:
                return None
            if "text/html" not in (r.headers.get("content-type") or ""):
                return None
            return await r.text(errors="ignore")
    except Exception:
        return None

async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> tuple[bytes | None, str]:
    """Return (bytes, content_type) or (None, '')."""
    try:
        async with session.get(url, timeout=TIMEOUT) as r:
            if r.status != 200:
                return None, ""
            ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
            data = await r.read()
            return data, ctype
    except Exception:
        return None, ""

# ---------- HTML parsing ----------
def parse_images_and_links(html: str, base_url: str) -> tuple[list[str], list[str]]:
    soup = make_soup(html)   # <-- was BeautifulSoup(html, "lxml")
    imgs, links = [], []
    ...

    # <img src> + srcset
    for img in soup.find_all("img"):
        if src := img.get("src"):
            imgs.append(urljoin(base_url, src))
        if srcset := img.get("srcset"):
            for part in SRCSET_SPLIT.split(srcset.strip()):
                cand = part.split()[0]
                if cand:
                    imgs.append(urljoin(base_url, cand))

    # <source srcset> (inside <picture>)
    for source in soup.find_all("source"):
        if srcset := source.get("srcset"):
            for part in SRCSET_SPLIT.split(srcset.strip()):
                cand = part.split()[0]
                if cand:
                    imgs.append(urljoin(base_url, cand))

    # inline CSS backgrounds
    for tag in soup.find_all(style=True):
        style = tag["style"]
        for m in IMG_URL_IN_STYLE.finditer(style):
            cand = m.group(2)
            if cand and not cand.startswith("data:"):
                imgs.append(urljoin(base_url, cand))

    # links to crawl
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href or href.startswith(("#","mailto:","javascript:")):
            continue
        links.append(urljoin(base_url, href))

    # dedupe & keep http(s)
    def uniq_http(lst):
        out = []
        seen = set()
        for u in lst:
            if urlsplit(u).scheme in {"http","https"} and u not in seen:
                seen.add(u); out.append(u)
        return out

    return uniq_http(imgs), uniq_http(links)

# ---------- robots.txt (very small) ----------
def parse_robots(text: str) -> dict:
    current_agents = []
    rules = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent:"):
            agent = line.split(":",1)[1].strip().lower()
            current_agents = [agent]
            rules.setdefault(agent, [])
        elif line.lower().startswith("disallow:"):
            path = line.split(":",1)[1].strip() or "/"
            for a in current_agents or ["*"]:
                rules.setdefault(a, []).append(path)
    return {"*": rules.get("*", [])}

async def get_robots(session: aiohttp.ClientSession, base: str) -> dict:
    if base in ROBOTS_CACHE:
        return ROBOTS_CACHE[base]
    url = base.rstrip("/") + "/robots.txt"
    try:
        async with session.get(url, timeout=TIMEOUT) as r:
            txt = await r.text(errors="ignore") if r.status == 200 else ""
    except Exception:
        txt = ""
    ROBOTS_CACHE[base] = parse_robots(txt)
    return ROBOTS_CACHE[base]

async def can_fetch(session: aiohttp.ClientSession, url: str, respect: bool) -> bool:
    if not respect:
        return True
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}"
    rules = await get_robots(session, base)
    path = parts.path or "/"
    for dis in rules.get("*", []):
        if dis and path.startswith(dis):
            return False
    return True

# ---------- Crawl core ----------
async def crawl_images(
    start_urls: list[str],
    depth: int,
    same_domain: bool,
    page_limit: int,
    max_images: int,
    min_bytes: int,
    verify_with_pillow: bool,
    respect_robots: bool,
    concurrency: int = 8,
    user_agent: str = DEFAULT_UA,
):
    connector = aiohttp.TCPConnector(limit_per_host=concurrency)
    sem = asyncio.Semaphore(concurrency)

    seen_pages: set[str] = set()
    seen_img_hashes: set[str] = set()
    images: list[dict] = []  # {url, data, mime, sha}

    q: asyncio.Queue = asyncio.Queue()
    for u in start_urls:
        await q.put((norm_url(u), 0))

    async with aiohttp.ClientSession(headers={"User-Agent": user_agent}, connector=connector) as session:
        async def worker():
            nonlocal images
            while len(images) < max_images:
                try:
                    url, d = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    return
                if url in seen_pages:
                    q.task_done(); continue
                if page_limit and len(seen_pages) >= page_limit:
                    q.task_done(); return

                if not await can_fetch(session, url, respect_robots):
                    seen_pages.add(url); q.task_done(); continue

                async with sem:
                    html = await fetch_text(session, url)
                seen_pages.add(url); q.task_done()
                if not html:
                    continue

                imgs, links = parse_images_and_links(html, url)

                # schedule image downloads (limited)
                for img_url in imgs:
                    if len(images) >= max_images:
                        break
                    # fetch bytes
                    async with sem:
                        data, mime = await fetch_bytes(session, img_url)
                    if not data or len(data) < min_bytes:
                        continue
                    if verify_with_pillow and HAVE_PIL:
                        try:
                            PILImage.open(io.BytesIO(data)).verify()
                        except Exception:
                            continue
                    digest = sha256(data)
                    if digest in seen_img_hashes:
                        continue
                    seen_img_hashes.add(digest)
                    images.append({"url": img_url, "data": data, "mime": mime, "sha": digest})

                # enqueue more pages
                if d < depth:
                    for link in links:
                        if same_domain and host_of(link) != host_of(url):
                            continue
                        if link not in seen_pages:
                            await q.put((link, d + 1))

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers, return_exceptions=True)

    return images  # list of dicts

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Image Scraper", page_icon="🖼️", layout="centered")
st.title("🖼️ Image Scraper")
st.caption("Paste a URL, choose depth, and fetch images (politely).")

with st.form("controls"):
    url = st.text_input("Start URL", placeholder="https://example.com")
    colA, colB = st.columns(2)
    with colA:
        depth = st.number_input("Crawl depth (0 = just this page)", 0, 4, 1, 1)
        same_domain = st.checkbox("Stay on same domain", value=True)
        respect_robots = st.checkbox("Respect robots.txt", value=True)
    with colB:
        max_pages = st.number_input("Max pages to crawl", 1, 1000, 150, 1)
        max_images = st.number_input("Max images to collect", 1, 1000, 200, 1)
        min_bytes = st.number_input("Min image size (bytes)", 0, 1_000_000, 8000, 1000)
    verify = st.checkbox("Verify images with Pillow (slower)", value=False)
    show_limit = st.slider("How many images to SHOW inline", 0, 200, 50, 5)
    submitted = st.form_submit_button("Scrape Images")

if submitted:
    if not url.strip():
        st.warning("Please enter a URL.")
        st.stop()

    # Run the crawler
    st.info("Scraping in progress…")
    with st.spinner("Fetching pages and images…"):
        images = asyncio.run(crawl_images(
            start_urls=[url],
            depth=int(depth),
            same_domain=bool(same_domain),
            page_limit=int(max_pages),
            max_images=int(max_images),
            min_bytes=int(min_bytes),
            verify_with_pillow=bool(verify),
            respect_robots=bool(respect_robots),
            concurrency=8,
        ))

    total = len(images)
    st.success(f"Found {total} image(s).")

    # Inline preview (first N as thumbnails)
    if show_limit and total:
        previews = [io.BytesIO(img["data"]).getvalue() for img in images[:show_limit]]
        caps = [urlsplit(img["url"]).path.rsplit("/", 1)[-1] or "(image)" for img in images[:show_limit]]
        st.image(previews, caption=caps, use_container_width=True)

    # Build a ZIP so the user can download
    if total:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for img in images:
                # pick an extension
                ext = ext_from_mime(img["mime"]) or ext_from_url(img["url"]) or ".bin"
                name = (img["sha"] + ext)
                zf.writestr(name, img["data"])
        buf.seek(0)
        st.download_button(
            label=f"⬇️ Download {total} image(s) as ZIP",
            data=buf,
            file_name="images.zip",
            mime="application/zip",
        )

st.caption("Be kind: limit depth/concurrency, and respect robots.txt when possible.")
