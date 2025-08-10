# house_hunter.py
# Requires:
#   pip install playwright
#   playwright install

import asyncio
import csv
import json
import os
import re
from typing import Dict, List, Set, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

from playwright.async_api import async_playwright, Page

SEARCH_URL = (
    "https://www.sreality.cz/hledani/prodej/domy"
    "?velikost=3-pokoje%2C4-pokoje%2C5-a-vice"
    "&cena-do=13000000"
    "&plocha-od=80"
    "&region=Zdiby"
    "&region-id=4227"
    "&region-typ=municipality"
    "&vzdalenost=5"
    "&pois_in_place_distance=2"
    "&pois_in_place=1"
)

OUTPUT_DIR = r"D:\#Projects & book notes\house_hunter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_JSON = os.path.join(OUTPUT_DIR, "sreality_zzdiby_listings.json")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "sreality_zzdiby_listings.csv")
BASE = "https://www.sreality.cz"

# ------------ text normalization ------------

ZERO_WIDTH = "".join([
    "\u200b", "\u200c", "\u200d", "\u200e", "\u200f",
    "\u2060", "\ufeff"
])
NBSP = "\xa0"
NNBSP = "\u202f"

def clean_text(s: str) -> str:
    if not s:
        return ""
    for ch in ZERO_WIDTH:
        s = s.replace(ch, "")
    s = s.replace(NBSP, " ").replace(NNBSP, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_space(s: str) -> str:
    return clean_text(s or "")

def kv_from_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in pairs:
        k = norm_space(k)
        v = norm_space(v)
        if not k and not v:
            continue
        if k in out and v:
            out[k] = f"{out[k]} | {v}"
        elif v:
            out[k] = v
        else:
            out.setdefault(k, "")
    return out

# ------------ safe getters ------------

async def get_first_text(page: Page, selectors: List[str], timeout_ms: int = 1000) -> str:
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if await loc.count():
                try:
                    txt = await loc.first.text_content(timeout=timeout_ms)
                    if txt:
                        return norm_space(txt)
                except:
                    pass
        except:
            pass
    return ""

async def get_first_inner_text(page: Page, selectors: List[str], timeout_ms: int = 1000) -> str:
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if await loc.count():
                try:
                    txt = await loc.first.inner_text(timeout=timeout_ms)
                    if txt:
                        return norm_space(txt)
                except:
                    pass
        except:
            pass
    return ""

# ------------ helpers ------------

def update_query_param(url: str, key: str, value: str) -> str:
    u = urlparse(url)
    q = parse_qs(u.query, keep_blank_values=True)
    q[key] = [str(value)]
    new_q = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))

async def click_first_visible(page_or_frame, selectors: List[str]) -> bool:
    for sel in selectors:
        try:
            loc = page_or_frame.locator(sel)
            if await loc.count():
                await loc.first.click(timeout=1500)
                await page_or_frame.wait_for_timeout(300)
                return True
        except:
            pass
    return False

async def handle_seznam_cmp(page: Page) -> None:
    accept_buttons = [
        'button:has-text("Souhlasím")',
        'button:has-text("Souhlasim")',
        'button:has-text("Přijmout vše")',
        'button:has-text("Prijmout vše")',
        'button:has-text("Prijmout vse")',
        'button:has-text("Rozumím")',
        '[data-testid="uc-accept-all-button"]',
        '#didomi-notice-agree-button',
        'button[aria-label*="accept" i]',
        'button:has-text("Accept all")',
    ]
    for _ in range(3):
        if "cmp.seznam.cz" in page.url:
            if await click_first_visible(page, accept_buttons):
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=15000)
                except:
                    pass
                await page.wait_for_timeout(500)
                continue
            else:
                for frame in page.frames:
                    if await click_first_visible(frame, accept_buttons):
                        try:
                            await page.wait_for_load_state("domcontentloaded", timeout=15000)
                        except:
                            pass
                        await page.wait_for_timeout(500)
                        break
                await page.wait_for_timeout(800)
        else:
            break
    await click_first_visible(page, accept_buttons)
    for frame in page.frames:
        await click_first_visible(frame, accept_buttons)

# ------------ scrolling ------------

async def gentle_scroll(page: Page, rounds: int = 8, pause_ms: int = 700) -> None:
    """Scroll to bottom a few times regardless of selectors (robust to A/B DOMs)."""
    for _ in range(rounds):
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        except:
            pass
        await page.wait_for_timeout(pause_ms)

# ------------ links on one results page (robust DOM) ------------

CARD_LINK_SELECTORS = [
    # Old layout:
    '[data-testid="result-list"] a[href^="/detail/prodej/dum/"]',
    '[data-testid="result-list"] a[href^="/detail/prodej/"]',
    # New layout (your sample):
    'ul[data-e2e="estates-list"] a[href^="/detail/prodej/dum/"]',
    'ul[data-e2e="estates-list"] a[href^="/detail/prodej/"]',
    # Generic fallback:
    'a[href^="/detail/prodej/dum/"]',
]

RESULTS_CONTAINER_SELECTORS = [
    '[data-testid="result-list"]',
    'ul[data-e2e="estates-list"]',
    '[data-e2e="estates-list"]',
]

async def collect_links_on_current_page(page: Page) -> List[str]:
    await handle_seznam_cmp(page)
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=15000)
    except:
        pass
    try:
        # Wait for either container variant (best-effort)
        await page.wait_for_selector(",".join(RESULTS_CONTAINER_SELECTORS), timeout=8000)
    except:
        pass

    # Scroll regardless of which DOM variant we get
    await gentle_scroll(page, rounds=8, pause_ms=700)

    links: Set[str] = set()
    for sel in CARD_LINK_SELECTORS:
        try:
            n = await page.locator(sel).count()
            for i in range(n):
                href = await page.locator(sel).nth(i).get_attribute("href")
                if not href:
                    continue
                if "/detail/prodej/dum/" in href or "/detail/prodej/" in href:
                    abs_url = urljoin(BASE, href.split("?")[0])
                    if "/detail/prodej/" in abs_url:
                        links.add(abs_url.replace("//detail", "/detail"))
        except:
            pass

    return sorted(links)

# ------------ crawl ALL pages deterministically (strana=N) ------------

def current_page_number_from_url(url: str) -> int:
    try:
        q = parse_qs(urlparse(url).query)
        return int(q.get("strana", ["1"])[0])
    except:
        return 1

async def collect_all_listing_links(page: Page, first_url: str) -> List[str]:
    seen_pages: Set[str] = set()
    all_links: Set[str] = set()

    cur_url = first_url
    page_idx = 1

    while True:
        if cur_url in seen_pages:
            break
        seen_pages.add(cur_url)

        print(f"Loading search page {page_idx}: {cur_url}")
        try:
            await page.goto(cur_url, wait_until="domcontentloaded", timeout=60000)
        except:
            await page.goto(cur_url, wait_until="domcontentloaded", timeout=60000)

        await handle_seznam_cmp(page)
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except:
            pass

        links_here = await collect_links_on_current_page(page)
        print(f"  -> found {len(links_here)} links on this page")
        if page_idx > 1 and len(links_here) == 0:
            # If a numbered page has no links, stop.
            print("  -> no links collected; stopping pagination.")
            break

        all_links.update(links_here)

        cur_n = current_page_number_from_url(cur_url)
        next_url = update_query_param(cur_url, "strana", str(cur_n + 1))
        if next_url in seen_pages:
            break

        cur_url = next_url
        page_idx += 1
        await page.wait_for_timeout(600)

    return sorted(all_links)

# ------------ detail extractor (incl. first image) ------------

async def extract_first_image_url(page: Page) -> str:
    img_selectors = [
        '[data-testid="gallery"] img',
        '[data-testid="image"] img',
        'figure img',
        'img[class*="image"]',
        'img[src*="sdn.cz"]',
        'img[srcset]',
        'img[data-src]',
    ]
    for sel in img_selectors:
        try:
            loc = page.locator(sel)
            if not await loc.count():
                continue
            el = loc.first
            # Try src / data-src
            for attr in ("src", "data-src"):
                try:
                    v = await el.get_attribute(attr)
                    if v and v.strip():
                        return urljoin(BASE, v.strip())
                except:
                    pass
            # Try srcset (pick the first)
            try:
                srcset = await el.get_attribute("srcset")
                if srcset:
                    first = srcset.split(",")[0].strip().split(" ")[0]
                    if first:
                        return urljoin(BASE, first)
            except:
                pass
        except:
            pass
    return ""

async def extract_key_values_from_detail(page: Page) -> Dict[str, str]:
    data: Dict[str, str] = {}

    data["title"] = await get_first_text(page, [
        'h1',
        '[data-testid="detail-title"]',
    ])

    data["locality"] = await get_first_text(page, [
        '[data-testid="location"]',
        'div[class*="location"]',
        'span[class*="location"]',
        'nav[aria-label="breadcrumb"] li:last-child',
        'div:has(> svg[aria-label="Place"])+div',
    ])

    data["price"] = await get_first_text(page, [
        '[data-testid="price"]',
        'div[class*="price"]',
        'span[class*="price"]',
    ])

    data["description"] = await get_first_inner_text(page, [
        'div[data-testid="detail-description"]',
        '[data-testid="description"]',
        'section:has(h2:has-text("Popis"))',
    ])

    # First image
    img_url = await extract_first_image_url(page)
    if img_url:
        data["image_url"] = img_url

    # Parameters (MUI dl/dt/dd)
    pairs: List[Tuple[str, str]] = []
    wrappers = [
        ".css-10q5btj",
        '[data-testid="params"]',
        "section:has(dt):has(dd)",
        "dl:has(dt):has(dd)",
    ]
    wrapper = None
    for sel in wrappers:
        try:
            if await page.locator(sel).count():
                wrapper = page.locator(sel).first
                break
        except:
            pass

    if wrapper:
        dls = wrapper.locator("dl")
        dl_count = await dls.count()
        for d in range(dl_count if dl_count else 1):
            scope = dls.nth(d) if dl_count else wrapper
            rows = scope.locator(":scope > div")
            row_count = await rows.count()
            if row_count == 0:
                dts = scope.locator("dt")
                dds = scope.locator("dd")
                for i in range(min(await dts.count(), await dds.count())):
                    try:
                        k = await dts.nth(i).inner_text(timeout=500)
                        v = await dds.nth(i).inner_text(timeout=500)
                        k = norm_space(k)
                        v = norm_space(", ".join(s.strip() for s in v.splitlines() if s.strip()))
                        if k and v:
                            pairs.append((k, v))
                    except:
                        pass
            else:
                for i in range(row_count):
                    row = rows.nth(i)
                    dt_loc = row.locator("dt")
                    dd_loc = row.locator("dd")
                    if await dt_loc.count() and await dd_loc.count():
                        try:
                            k = await dt_loc.first.inner_text(timeout=500)
                            raw = await dd_loc.first.inner_text(timeout=500)
                            k = norm_space(k)
                            v = norm_space(", ".join(s.strip() for s in raw.splitlines() if s.strip()))
                            if k and v:
                                pairs.append((k, v))
                        except:
                            pass

    if not data.get("price"):
        for k, v in pairs:
            if "celková cena" in k.lower():
                data["price"] = v
                break

    # ID fallback
    try:
        html = await page.content()
        m = re.search(r'(ID|Evidenční číslo|ID zakázky)\s*[:#]?\s*([0-9]{4,})', html, re.IGNORECASE)
        if m and not any(k.lower().startswith("id") for k, _ in pairs):
            pairs.append(("ID zakázky", m.group(2)))
    except:
        pass

    # GPS if present
    try:
        html = await page.content()
        m = re.search(r'"lat"\s*:\s*([0-9]+\.[0-9]+)\s*,\s*"lon"\s*:\s*([0-9]+\.[0-9]+)', html)
        if m:
            data["latitude"] = m.group(1)
            data["longitude"] = m.group(2)
    except:
        pass

    data.update(kv_from_pairs(pairs))

    try:
        data["url"] = page.url.split("?")[0]
    except:
        pass

    data = {k: v for k, v in data.items() if v}
    return data

# ------------ main ------------

async def run():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(
            viewport={"width": 1366, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        page.set_default_timeout(5000)

        print("Opening search URL…")
        await page.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=60000)
        await handle_seznam_cmp(page)
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except:
            pass

        print("Discovering ALL result pages and collecting links…")
        links = await collect_all_listing_links(page, SEARCH_URL)
        print(f"Total unique listing links found: {len(links)}")

        results: List[Dict[str, str]] = []

        for idx, link in enumerate(links, start=1):
            print(f"[{idx}/{len(links)}] Visiting: {link}")
            try:
                await page.goto(link, wait_until="domcontentloaded", timeout=60000)
            except:
                try:
                    await page.goto(link, wait_until="domcontentloaded", timeout=60000)
                except Exception as e:
                    print(f"  -> Failed to open: {e}")
                    continue

            await handle_seznam_cmp(page)
            await page.wait_for_timeout(5000)  # required delay

            # Expand “show more” if present (best-effort)
            for sel in [
                'button:has-text("Zobrazit více")',
                'button:has-text("Více informací")',
                'button[aria-expanded="false"]',
            ]:
                try:
                    if await page.locator(sel).count():
                        await page.locator(sel).first.click(timeout=800)
                        await page.wait_for_timeout(200)
                except:
                    pass

            data = await extract_key_values_from_detail(page)
            if data:
                results.append(data)

        await context.close()
        await browser.close()

    # Save
    if results:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        all_keys: Set[str] = set()
        for r in results:
            all_keys.update(r.keys())
        cols = sorted(all_keys)

        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in results:
                w.writerow(r)

        print(f"\nSaved {len(results)} listings to:")
        print(f"  - JSON: {os.path.abspath(OUTPUT_JSON)}")
        print(f"  - CSV : {os.path.abspath(OUTPUT_CSV)}")
    else:
        print("No results extracted. Try printing one detail page's HTML to tune selectors.")

if __name__ == "__main__":
    asyncio.run(run())
