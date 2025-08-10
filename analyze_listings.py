# analyze_listings.py
# Streamlit dashboard for Sreality listings
# Run: streamlit run analyze_listings.py
# pip install streamlit pandas altair pydeck

import json
import math
import os
import re
import unicodedata
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk

# ---------- Page setup ----------
st.set_page_config(page_title="House Hunter – Sreality", page_icon="🏠", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
.vega-img-tip { z-index: 99999; }
</style>
""", unsafe_allow_html=True)

# ---------- Constants ----------
DEFAULT_JSON = "sreality_zzdiby_listings.json"
DEFAULT_WORK_LAT = 50.11137098418734
DEFAULT_WORK_LON = 14.440321219190835
EMBED_WIDTH = 1280  # width of the iframe used to host each chart

# ---------- Text normalization ----------
ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u2060", "\ufeff"]
NBSP  = "\xa0"
NNBSP = "\u202f"

def clean_text(s: str) -> str:
    if not s: return ""
    for ch in ZERO_WIDTH:
        s = s.replace(ch, "")
    s = s.replace(NBSP, " ").replace(NNBSP, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def parse_price_czk(s: str) -> float:
    s = clean_text(s or "")
    s = re.sub(r"[^\d,\.]", "", s).replace(",", "").replace(".", "")
    return float(s) if s else np.nan

def num_from_area_token(token: str) -> float:
    token = clean_text(token)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*m", token.lower()) or re.search(r"(\d+(?:[.,]\d+)?)", token)
    if not m: return np.nan
    try:
        return float(m.group(1).replace(",", "."))
    except Exception:
        return np.nan

def parse_plochy_block(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    s = clean_text(s or "")
    if not s: return out
    for p in re.split(r",|\n", s):
        p = p.strip()
        if not p: continue
        lower = p.lower()
        val = num_from_area_token(p)
        if "pozemku" in lower:
            out["plocha_pozemku_m2"] = val
        elif "užitná" in lower or "uzitna" in lower:
            out["uzitna_plocha_m2"] = val
        elif "zastavěná" in lower or "zastavena" in lower:
            out["zastavena_plocha_m2"] = val
        elif "celková" in lower or "celkova" in lower:
            out["celkova_plocha_m2"] = val
    return out

def plochy_from_row(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ["Plocha:", "Plocha", "Plocha :"]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            out.update(parse_plochy_block(row[key]))
            break
    mapping = {
        "Plocha pozemku": "plocha_pozemku_m2",
        "Užitná plocha": "uzitna_plocha_m2",
        "Zastavěná plocha": "zastavena_plocha_m2",
        "Celková plocha": "celkova_plocha_m2",
    }
    for cz, std in mapping.items():
        for k in [cz, f"{cz}:"]:
            if k in row and isinstance(row[k], str) and row[k].strip():
                out[std] = num_from_area_token(row[k])
    return out

# ---------- Geo ----------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ---------- Data loading / enrichment ----------
@st.cache_data(show_spinner=False)
def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def enrich_dataframe(records: List[dict], work_lat: float, work_lon: float) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "url" not in df.columns:
        df["url"] = ""

    # label
    if "title" in df.columns:
        df["label"] = df["title"].apply(lambda x: clean_text(str(x))[:80] if pd.notna(x) else "")
    elif "url" in df.columns:
        df["label"] = df["url"]
    else:
        df["label"] = [f"listing_{i}" for i in range(len(df))]

    # price
    price_cols = ["price", "Celková cena:", "Celková cena"]
    df["price_czk"] = np.nan
    for i, row in df.iterrows():
        raw = ""
        for c in price_cols:
            if c in df.columns and pd.notna(row.get(c)):
                raw = str(row[c]); break
        df.at[i, "price_czk"] = parse_price_czk(raw)

    # areas
    for c in ["plocha_pozemku_m2", "uzitna_plocha_m2", "zastavena_plocha_m2", "celkova_plocha_m2"]:
        df[c] = np.nan
    for i, row in df.iterrows():
        for k, v in plochy_from_row(row).items():
            df.at[i, k] = v

    # price per sqm
    def q(a, b):
        try:
            a = float(a); b = float(b)
            return a / b if (not pd.isna(b) and b != 0) else np.nan
        except Exception:
            return np.nan

    df["ppsqm_uzitna"]    = [q(a, b) for a, b in zip(df["price_czk"], df["uzitna_plocha_m2"])]
    df["ppsqm_pozemek"]   = [q(a, b) for a, b in zip(df["price_czk"], df["plocha_pozemku_m2"])]
    df["ppsqm_zastavena"] = [q(a, b) for a, b in zip(df["price_czk"], df["zastavena_plocha_m2"])]
    df["ppsqm_celkova"]   = [q(a, b) for a, b in zip(df["price_czk"], df["celkova_plocha_m2"])]

    # distance to work
    def to_f(x):
        try: return float(x)
        except Exception: return np.nan
    lat = df.get("latitude", pd.Series([None] * len(df))).apply(to_f)
    lon = df.get("longitude", pd.Series([None] * len(df))).apply(to_f)
    df["distance_to_work_km"] = [
        haversine_km(la, lo, work_lat, work_lon) if not (pd.isna(la) or pd.isna(lo)) else np.nan
        for la, lo in zip(lat, lon)
    ]
    return df

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Data & Settings")
default_path = DEFAULT_JSON if os.path.exists(DEFAULT_JSON) else ""
json_path = st.sidebar.text_input("JSON file path", value=default_path, help="Path to the scraper output JSON.")
uploaded = st.sidebar.file_uploader("...or upload JSON", type=["json"])

st.sidebar.markdown("---")
work_lat = st.sidebar.number_input("Work latitude", value=float(DEFAULT_WORK_LAT), format="%.8f")
work_lon = st.sidebar.number_input("Work longitude", value=float(DEFAULT_WORK_LON), format="%.8f")

area_choice = st.sidebar.selectbox(
    "Area basis for price per m²",
    options=[
        ("ppsqm_uzitna", "užitná plocha"),
        ("ppsqm_pozemek", "plocha pozemku"),
        ("ppsqm_zastavena", "zastavěná plocha"),
        ("ppsqm_celkova", "celková plocha"),
    ],
    format_func=lambda x: x[1], index=0
)

# NEW: toggle to hide listings "Před rekonstrukcí" (from Stavba / Stav objektu / Stav)
hide_pred_rek = st.sidebar.checkbox('Hide listings marked "Před rekonstrukcí"', value=False,
                                    help="Filters out listings whose condition states 'Před rekonstrukcí'.")

st.sidebar.markdown("### Weighted ranking (lower = better)")
w_ppsqm = st.sidebar.slider("Weight: price per m²", 0.0, 1.0, 0.7, 0.05)
w_dist  = 1.0 - w_ppsqm
st.sidebar.caption(f"Distance weight auto-set to {w_dist:.2f}")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use the table filters to subset before downloading.")

# ---------- Load data ----------
records = None
try:
    if uploaded is not None:
        records = json.load(uploaded)
    elif json_path:
        records = load_json(json_path)
except Exception as e:
    st.error(f"Failed to load JSON: {e}")

if not records:
    st.info("Load your scraper JSON using the sidebar to begin.")
    st.stop()

df = enrich_dataframe(records, work_lat, work_lon)

# ---------- Derived / filters ----------
ppsqm_col = area_choice[0]

def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty: return s
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.5 if not pd.isna(x) else np.nan for x in s], index=s.index)
    return (s - mn) / (mx - mn)

df["rank_metric_ppsqm"] = minmax(df[ppsqm_col])
df["rank_metric_dist"]  = minmax(df["distance_to_work_km"])
df["score"] = w_ppsqm * df["rank_metric_ppsqm"] + w_dist * df["rank_metric_dist"]

# Build a mask for "Před rekonstrukcí" across plausible fields
stav_cols = [c for c in df.columns if any(k in c.lower() for k in ["stavba", "stav objektu", "stav"])]
def row_is_pred_rek(row) -> bool:
    for c in stav_cols:
        v = clean_text(str(row.get(c, ""))).lower()
        if not v: continue
        if "před rekonstrukcí" in v:  # diacritics version
            return True
        v_ascii = ascii_fold(v)
        if "pred rekonstrukci" in v_ascii:  # folded version
            return True
    return False

if stav_cols:
    df["is_pred_rekonstrukci"] = df.apply(row_is_pred_rek, axis=1)
else:
    df["is_pred_rekonstrukci"] = False

def slider_range(series: pd.Series, label: str, step: float = 1.0):
    s = series.dropna().astype(float)
    if s.empty: return None
    lo, hi = float(np.floor(s.min())), float(np.ceil(s.max()))
    if lo == hi:
        st.sidebar.caption(f"{label}: {lo} (fixed)")
        return (lo, hi)
    return st.sidebar.slider(label, min_value=lo, max_value=hi, value=(lo, hi), step=step)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
price_rng = slider_range(df["price_czk"], "Price (CZK)", step=10000.0)
dist_rng  = slider_range(df["distance_to_work_km"], "Distance to work (km)", step=0.5)
ppsqm_rng = slider_range(df[ppsqm_col], f"{ppsqm_col} (CZK/m²)", step=100.0)

mask = pd.Series([True] * len(df))
if price_rng:
    mask &= df["price_czk"].between(price_rng[0], price_rng[1], inclusive="both")
if dist_rng:
    mask &= df["distance_to_work_km"].between(dist_rng[0], dist_rng[1], inclusive="both")
if ppsqm_rng:
    mask &= df[ppsqm_col].between(ppsqm_rng[0], ppsqm_rng[1], inclusive="both")
if hide_pred_rek:
    mask &= ~df["is_pred_rekonstrukci"].fillna(False)

df_view = df[mask].copy()
df_view.sort_values("score", inplace=True)

# ---------- Title / KPIs ----------
st.title("🏠 House Hunter – Sreality analytics")
st.caption("Click any bar/point/heatmap cell to open the ad in a new browser tab.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings (shown / total)", f"{len(df_view)}/{len(df)}")
c2.metric("Median price", f"{int(np.nanmedian(df_view['price_czk'])):,} Kč".replace(",", " ") if not df_view['price_czk'].dropna().empty else "—")
c3.metric(f"Median {ppsqm_col}", f"{int(np.nanmedian(df_view[ppsqm_col])):,} Kč/m²".replace(",", " ") if not df_view[ppsqm_col].dropna().empty else "—")
c4.metric("Median distance", f"{np.nanmedian(df_view['distance_to_work_km']):.1f} km" if not df_view['distance_to_work_km'].dropna().empty else "—")

# ---------- Helper: embed Altair that fits & opens in new tab ----------
def render_altair_click_newtab(chart: alt.Chart, height: int, key: Optional[str] = None):
    spec = chart.to_dict()
    spec.pop("width", None)  # let JS fit width
    div_id = f"vis-{(key or uuid4().hex)}"

    base_html = r"""
    <div id="__DIVID__" style="width:100%"></div>
    <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    <script>
      const container = document.getElementById("__DIVID__");
      const spec = __SPECJSON__;

      function draw() {
        const w = Math.max(320, container.clientWidth);
        const s = Object.assign({}, spec, {
          width: w,
          autosize: { type: "fit", contains: "padding", resize: true }
        });
        vegaEmbed("#__DIVID__", s, { actions: false, renderer: "svg" }).then(function(res) {
          const view = res.view;
          view.addEventListener("click", function(event, item) {
            if (item && item.datum && item.datum.url) {
              window.open(item.datum.url, "_blank", "noopener");
            }
          });
          const svg = container.querySelector("svg");
          if (svg) svg.style.cursor = "pointer";
        });
      }

      draw();
      new ResizeObserver(() => draw()).observe(container);
      window.addEventListener("resize", draw);
    </script>
    """
    html = base_html.replace("__DIVID__", div_id).replace("__SPECJSON__", json.dumps(spec))
    st.components.v1.html(html, height=height, width=EMBED_WIDTH, scrolling=False)

# ---------- Charts ----------
st.subheader("Price per m² ranking")
rank_df = df_view[["label", ppsqm_col, "score", "url"]].dropna(subset=[ppsqm_col]).sort_values(ppsqm_col, ascending=True)
if not rank_df.empty:
    bar = (
        alt.Chart(rank_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{ppsqm_col}:Q", title=f"{ppsqm_col} (CZK/m²)"),
            y=alt.Y("label:N", sort="-x", title=""),
            tooltip=["label", alt.Tooltip(f"{ppsqm_col}:Q", format=",.0f"), alt.Tooltip("score:Q", format=".2f"), "url:N"],
        )
        .properties(height=min(30 * len(rank_df), 900))
    )
    render_altair_click_newtab(bar, height=min(30 * len(rank_df), 900), key="bar")
else:
    st.info("No data for price-per-m² ranking.")

st.subheader("Distance vs. price per m²")
scatter_df = df_view[["distance_to_work_km", ppsqm_col, "label", "url"]].dropna(subset=[ppsqm_col, "distance_to_work_km"])
if not scatter_df.empty:
    scatter = (
        alt.Chart(scatter_df)
        .mark_point()
        .encode(
            x=alt.X("distance_to_work_km:Q", title="Distance to work (km)"),
            y=alt.Y(f"{ppsqm_col}:Q", title=f"{ppsqm_col} (CZK/m²)"),
            tooltip=["label", alt.Tooltip("distance_to_work_km:Q", format=".2f"), alt.Tooltip(f"{ppsqm_col}:Q", format=",.0f"), "url:N"],
        )
        .interactive()
        .properties(height=420)
    )
    render_altair_click_newtab(scatter, height=420, key="scatter")
else:
    st.info("No data for scatter plot.")

st.subheader("Comparison heatmap (normalized)")
heat_cols = ["ppsqm_uzitna", "ppsqm_pozemek", "ppsqm_zastavena", "distance_to_work_km"]
avail_heat_cols = [c for c in heat_cols if c in df_view.columns and not df_view[c].dropna().empty]
if avail_heat_cols:
    H = df_view[["label", "url"] + avail_heat_cols].dropna()
    if not H.empty:
        for c in avail_heat_cols:
            H[c] = minmax(H[c])
        Hm = H.melt(id_vars=["label", "url"], var_name="metric", value_name="normalized")
        heat = (
            alt.Chart(Hm)
            .mark_rect()
            .encode(
                y=alt.Y("label:N", sort="-x", title=""),
                x=alt.X("metric:N", title="Metric"),
                color=alt.Color("normalized:Q", title="normalized"),
                tooltip=["label", "metric", alt.Tooltip("normalized:Q", format=".2f"), "url:N"],
            )
            .properties(height=min(30 * len(H["label"].unique()), 900))
        )
        render_altair_click_newtab(heat, height=min(30 * len(H["label"].unique()), 900), key="heatmap")
    else:
        st.info("No complete rows for heatmap.")
else:
    st.info("No metrics available for heatmap.")

# ---------- Map ----------
st.subheader("Map")
map_df = df_view[["label", "latitude", "longitude", "price_czk", ppsqm_col, "url"]].copy()
map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
map_df.dropna(subset=["latitude", "longitude"], inplace=True)

if not map_df.empty:
    if map_df[ppsqm_col].notna().any():
        if map_df[ppsqm_col].nunique() > 1:
            mm = (map_df[ppsqm_col] - map_df[ppsqm_col].min()) / (map_df[ppsqm_col].max() - map_df[ppsqm_col].min())
        else:
            mm = pd.Series([0.5] * len(map_df), index=map_df.index)
    else:
        mm = pd.Series([0.5] * len(map_df), index=map_df.index)
    map_df["norm"] = mm.fillna(0.5)

    base1 = np.array([80, 120, 200])
    base2 = np.array([200, 80, 80])
    colors = (base1[None, :] * (1 - map_df["norm"].values[:, None]) + base2[None, :] * map_df["norm"].values[:, None]).astype(int)
    map_df["r"], map_df["g"], map_df["b"] = colors[:, 0], colors[:, 1], colors[:, 2]

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_radius=30,
        get_fill_color="[r, g, b, 160]",
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=float(map_df["latitude"].mean()),
        longitude=float(map_df["longitude"].mean()),
        zoom=11,
        pitch=0,
    )
    tooltip = {
        "html": "<b>{label}</b><br/>Price: {price_czk}<br/>" + ppsqm_col + ": {" + ppsqm_col + "}"
                "<br/><a href='{url}' target='_blank'>Open ad ↗</a>",
        "style": {"backgroundColor": "white", "color": "black"}
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
else:
    st.info("No geocoded listings to map.")

# ---------- Table & Download ----------
st.subheader("Table")
def dataframe_compat(df_in: pd.DataFrame, use_container_width=True):
    try:
        return st.dataframe(df_in, use_container_width=use_container_width, hide_index=True)
    except TypeError:
        return st.dataframe(df_in.reset_index(drop=True), use_container_width=use_container_width)

show_cols = [
    "label", "price_czk", "distance_to_work_km",
    "uzitna_plocha_m2", "plocha_pozemku_m2", "zastavena_plocha_m2", "celkova_plocha_m2",
    "ppsqm_uzitna", "ppsqm_pozemek", "ppsqm_zastavena", "ppsqm_celkova",
    "score", "url", "is_pred_rekonstrukci"
]
show_cols = [c for c in show_cols if c in df_view.columns]
dataframe_compat(df_view[show_cols], use_container_width=True)

st.markdown("#### Download enriched CSV")
csv_bytes = df_view.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="listings_enriched_filtered.csv", mime="text/csv")
