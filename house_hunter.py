# house_hunter.py
# Requires: pip install playwright
# Then one-time: playwright install

import asyncio
import csv
import json
import os
import re
from typing import Dict, List, Set
from urllib.parse import urljoin

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
    "\u200b", "\u200c", "\u200d", "\u200e", "\u200f",  # zero-width + bidi
    "\u2060", "\ufeff"                                 # word joiner + BOM
])
NBSP = "\xa0"           # nbsp
NNBSP = "\u202f"        # narrow nbsp

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

def kv_from_pairs(pairs: List[tuple[str, str]]) -> Dict[str, str]:
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
    """Return text of the first existing selector; never throws."""
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
    """Return innerText of the first existing selector; never throws."""
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

# ------------ CMP handling ------------

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

# ------------ listings collection ------------

async def infinite_scroll(page: Page, item_selector: str, max_rounds: int = 40) -> None:
    prev = 0
    stagnation = 0
    for _ in range(max_rounds):
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        except:
            pass
        await page.wait_for_timeout(900)
        try:
            count = await page.locator(item_selector).count()
        except:
            count = 0
        if count <= prev:
            stagnation += 1
            if stagnation >= 3:
                break
        else:
            stagnation = 0
            prev = count

async def collect_listing_links(page: Page) -> List[str]:
    card_selector = '[data-testid="result-list"] a[href^="/detail/prodej/dum/"]'
    fallback_selector = 'a[href^="/detail/prodej/dum/"]'

    await handle_seznam_cmp(page)
    try:
        await page.wait_for_load_state("networkidle", timeout=10000)
    except:
        pass

    await infinite_scroll(page, item_selector=card_selector, max_rounds=40)

    links: Set[str] = set()
    for sel in (card_selector, fallback_selector):
        try:
            n = await page.locator(sel).count()
            for i in range(n):
                href = await page.locator(sel).nth(i).get_attribute("href")
                if href and "/detail/prodej/dum/" in href:
                    links.add(urljoin(BASE, href.split("?")[0]))
        except:
            pass
    return sorted({l.replace("//detail", "/detail") for l in links})

# ------------ detail extractor (robust to your structure) ------------

async def extract_key_values_from_detail(page: Page) -> Dict[str, str]:
    data: Dict[str, str] = {}

    # Title
    data["title"] = await get_first_text(page, [
        'h1',
        '[data-testid="detail-title"]',
    ])

    # Locality — try multiple likely spots; never wait long
    data["locality"] = await get_first_text(page, [
        '[data-testid="location"]',
        'div[class*="location"]',
        'span[class*="location"]',
        'nav[aria-label="breadcrumb"] li:last-child',
        'div:has(> svg[aria-label="Place"])+div',  # sometimes an icon + text pattern
    ])

    # Price — try big price, then param fallback
    data["price"] = await get_first_text(page, [
        '[data-testid="price"]',
        'div[class*="price"]',
        'span[class*="price"]',
    ])

    # Description (best-effort)
    data["description"] = await get_first_inner_text(page, [
        'div[data-testid="detail-description"]',
        '[data-testid="description"]',
        'section:has(h2:has-text("Popis"))',
    ])

    # Parameters (MUI div-wrapped <dt>/<dd> pairs)
    pairs: List[tuple[str, str]] = []
    wrappers = [
        ".css-10q5btj",            # from your sample
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
        # Every dl in wrapper
        dls = wrapper.locator("dl")
        dl_count = await dls.count()
        for d in range(dl_count if dl_count else 1):
            scope = dls.nth(d) if dl_count else wrapper
            # rows like: <div> <dt>...</dt> <dd>...</dd> </div>
            rows = scope.locator(":scope > div")
            row_count = await rows.count()
            if row_count == 0:
                # fallback to any dt in scope
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

    # If price still empty, try the "Celková cena:" param
    if not data.get("price"):
        for k, v in pairs:
            if "celková cena" in k.lower():
                data["price"] = v
                break

    # ID fallback from full HTML (in case not exposed as a pair)
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

    # Canonical URL
    try:
        data["url"] = page.url.split("?")[0]
    except:
        pass

    # Strip empties
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
        # Keep Playwright from waiting forever on any single call
        page.set_default_timeout(5000)

        print("Opening search URL…")
        await page.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=60000)
        await handle_seznam_cmp(page)
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except:
            pass

        print("Loading all listings (scrolling)…")
        links = await collect_listing_links(page)
        if not links:
            await infinite_scroll(page, item_selector='a[href^="/detail/prodej/dum/"]', max_rounds=40)
            links = await collect_listing_links(page)

        print(f"Found {len(links)} listing links.")
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

            # Expand “show more” if present (best-effort; non-blocking)
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
