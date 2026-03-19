"""
website_crawler.py - Async website crawler
Extracts clean content from university websites for the knowledge base
Uses trafilatura for main-content extraction (ignores menus/footers)
"""

import asyncio
import logging
import time
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set, Optional

import httpx

logger = logging.getLogger(__name__)


# ─── CONTENT EXTRACTION ──────────────────────────────────────

def extract_main_content(html: str, url: str) -> Optional[str]:
    """
    Extract main content from HTML using trafilatura.
    Automatically removes navigation, ads, footers, cookie banners.
    Falls back to BeautifulSoup if trafilatura returns nothing.
    """
    # Try trafilatura first (best quality)
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if text and len(text.strip()) > 50:  # Lower threshold
            return text.strip()
    except ImportError:
        logger.warning("trafilatura not installed, falling back to BeautifulSoup")
    except Exception as e:
        logger.warning(f"trafilatura failed for {url}: {e}")

    # Fallback: BeautifulSoup
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "iframe", "noscript", "form",
                         ".cookie-banner", ".advertisement"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up
        lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 10]  # Lower threshold
        return "\n".join(lines) if lines else None

    except ImportError:
        logger.error("Neither trafilatura nor beautifulsoup4 is installed")
        return None
    except Exception as e:
        logger.error(f"BeautifulSoup fallback failed for {url}: {e}")
        return None


def extract_links(html: str, base_url: str) -> List[str]:
    """Extract all internal links from a page."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(base_url).netloc
        links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Build absolute URL
            absolute_url = urljoin(base_url, href)
            parsed = urlparse(absolute_url)

            # Only follow links on the same domain
            if (
                parsed.netloc == base_domain
                and parsed.scheme in ("http", "https")
                and "#" not in absolute_url       # skip anchors
                and "?" not in absolute_url       # skip query strings (optional)
                and not absolute_url.endswith(    # skip file downloads
                    (".pdf", ".docx", ".xlsx", ".zip", ".png", ".jpg", ".jpeg")
                )
            ):
                # Normalize URL (remove trailing slash)
                clean_url = absolute_url.rstrip("/")
                links.append(clean_url)

        return list(set(links))  # deduplicate

    except Exception as e:
        logger.error(f"Link extraction failed: {e}")
        return []


# ─── ROBOTS.TXT ──────────────────────────────────────────────

async def check_robots_allowed(url: str, client: httpx.AsyncClient) -> bool:
    """
    Check robots.txt to be a respectful crawler.
    Returns True if crawling is allowed, False if disallowed.
    """
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        response = await client.get(robots_url, timeout=5)
        if response.status_code != 200:
            return True  # No robots.txt = allowed

        robots_txt = response.text.lower()

        # Very basic check: look for Disallow: / for all user agents
        lines = robots_txt.split("\n")
        in_our_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("user-agent:"):
                agent = line.split(":", 1)[1].strip()
                in_our_section = agent in ("*", "python-httpx")
            elif in_our_section and line.startswith("disallow:"):
                disallowed = line.split(":", 1)[1].strip()
                if disallowed == "/":  # entire site disallowed
                    return False

        return True

    except Exception:
        return True  # If we can't check, assume allowed


# ─── ASYNC PAGE FETCHER ──────────────────────────────────────

async def fetch_page(
    url: str,
    client: httpx.AsyncClient,
    delay: float = 1.0,
    timeout: int = 10,
) -> Optional[Dict]:
    """
    Fetch a single page and extract its content.
    Returns dict with url, title, content or None on failure.
    """
    await asyncio.sleep(delay)  # Be respectful — don't hammer servers

    try:
        headers = {
            "User-Agent": "UniversityBot/1.3 (Educational chatbot; respectful crawler)",
            "Accept": "text/html,application/xhtml+xml",
        }

        response = await client.get(url, headers=headers, timeout=timeout,
                                    follow_redirects=True)

        # Only process HTML pages
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None

        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} for {url}")
            return None

        html = response.text

        # Extract content
        content = extract_main_content(html, url)
        if not content or len(content) < 50:  # Lower threshold
            logger.debug(f"Skipping {url} — insufficient content")
            return None

        # Try to get page title
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.string.strip() if soup.title else url
        except Exception:
            title = url

        # Prepend URL and title for context
        full_content = f"Source: {url}\nTitle: {title}\n\n{content}"

        return {
            "url":     url,
            "title":   title,
            "content": full_content,
            "html":    html,
        }

    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching: {url}")
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


# ─── MAIN CRAWLER ────────────────────────────────────────────

async def crawl_website(
    start_url: str,
    max_pages: int = 20,
    delay: float = 1.0,
) -> Dict:
    """
    Crawl a website starting from start_url.
    Stays within the same domain.
    Returns all extracted content as a combined string + stats.

    Args:
        start_url: URL to start crawling from
        max_pages: Maximum number of pages to visit
        delay: Seconds to wait between requests

    Returns:
        {
            "combined_text": str,   # all page content joined
            "pages_crawled": int,
            "pages_found": int,
            "failed_pages": int,
            "sources": list of {"url", "title"}
        }
    """
    logger.info(f"Starting crawl: {start_url} (max={max_pages} pages)")
    start_time = time.time()

    visited: Set[str] = set()
    to_visit: List[str] = [start_url.rstrip("/")]
    pages_content: List[str] = []
    sources: List[Dict] = []
    failed_count = 0

    async with httpx.AsyncClient() as client:

        # Check robots.txt first
        allowed = await check_robots_allowed(start_url, client)
        if not allowed:
            logger.warning(f"robots.txt disallows crawling {start_url}")
            return {
                "combined_text": "",
                "pages_crawled": 0,
                "pages_found":   0,
                "failed_pages":  0,
                "sources":       [],
                "error":         "robots.txt disallows crawling this site",
            }

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)

            if url in visited:
                continue

            visited.add(url)
            logger.info(f"Crawling [{len(visited)}/{max_pages}]: {url}")

            # Fetch and extract content
            page = await fetch_page(url, client, delay=delay)

            if page:
                pages_content.append(page["content"])
                sources.append({"url": page["url"], "title": page["title"]})

                # Discover new links on this page
                new_links = extract_links(page["html"], url)
                for link in new_links:
                    if link not in visited and link not in to_visit:
                        to_visit.append(link)

                logger.debug(f"Found {len(new_links)} links on {url}")
            else:
                failed_count += 1

    elapsed = round(time.time() - start_time, 1)
    pages_crawled = len(pages_content)

    logger.info(
        f"Crawl complete: {pages_crawled} pages in {elapsed}s "
        f"({failed_count} failed, {len(to_visit)} not visited)"
    )

    return {
        "combined_text": "\n\n" + ("=" * 60) + "\n\n".join(pages_content),
        "pages_crawled": pages_crawled,
        "pages_found":   len(visited),
        "failed_pages":  failed_count,
        "sources":       sources,
        "elapsed_sec":   elapsed,
    }
