# analyze_listings.py
# Streamlit dashboard for Sreality listings
# Run: streamlit run analyze_listings.py
# Recommended: pip install streamlit pandas altair pydeck

import json
import math
import os
import re
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import pydeck as pdk

# ---------- Page setup ----------
st.set_page_config(
    page_title="House Hunter – Sreality",
    page_icon="🏠",
    layout="wide",
)

st.markdown("""
<style>
/* Make the page a bit tighter and cleaner */
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Defaults ----------
DEFAULT_JSON = "sreality_zzdiby_listings.json"
DEFAULT_WORK_LAT = 50.20369338989258
DEFAULT_WORK_LON = 14.475787162780762

# ---------- Text normalization ----------
ZERO_WIDTH = [
    "\u200b", "\u200c", "\u200d", "\u200e", "\u200f",  # zero-width + bidi
    "\u2060", "\ufeff"                                 # word joiner + BOM
]
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

def parse_price_czk(s: str) -> float:
    """ Convert '10 500 000 Kč' (with weird spaces) -> 10500000.0 """
    s = clean_text(s or "")
    s = re.sub(r"[^\d,\.]", "", s)  # keep digits and separators
    s = s.replace(",", "").replace(".", "")
    return float(s) if s else np.nan

def num_from_area_token(token: str) -> float:
    """ Extract number from 'Užitná plocha 105 m²' -> 105 """
    token = clean_text(token)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*m", token.lower())
    if not m:
        m = re.search(r"(\d+(?:[.,]\d+)?)", token)
    if m:
        val = m.group(1).replace(",", ".")
        try:
            return float(val)
        except:
            return np.nan
    return np.nan

def parse_plochy_block(s: str) -> Dict[str, float]:
    """
    Parse combined 'Plocha:' like:
    'Plocha pozemku 556 m², Užitná plocha 105 m², Zastavěná plocha 73 m², Celková plocha 105 m²'
    """
    out: Dict[str, float] = {}
    s = clean_text(s or "")
    if not s:
        return out
    parts = re.split(r",|\n", s)
    for p in parts:
        p = p.strip()
        if not p:
            continue
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
    """
    Pull areas from either a combined 'Plocha:' key or separate keys.
    """
    out: Dict[str, float] = {}
    # Combined
    for key in ["Plocha:", "Plocha", "Plocha :"]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            out.update(parse_plochy_block(row[key]))
            break
    # Separate keys
    sep_map = {
        "Plocha pozemku": "plocha_pozemku_m2",
        "Užitná plocha": "uzitna_plocha_m2",
        "Zastavěná plocha": "zastavena_plocha_m2",
        "Celková plocha": "celkova_plocha_m2",
    }
    for k_cz, k_std in sep_map.items():
        for kk in [k_cz, f"{k_cz}:"]:
            if kk in row and isinstance(row[kk], str) and row[kk].strip():
                out[k_std] = num_from_area_token(row[kk])
    return out

# ---------- Geo ----------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ---------- Data loading / enrichment ----------
@st.cache_data(show_spinner=False)
def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def enrich_dataframe(records: List[dict], work_lat: float, work_lon: float) -> pd.DataFrame:
    df = pd.DataFrame(records)

    # Label
    if "title" in df.columns:
        df["label"] = df["title"].apply(lambda x: clean_text(str(x))[:80] if pd.notna(x) else "")
    elif "url" in df.columns:
        df["label"] = df["url"]
    else:
        df["label"] = [f"listing_{i}" for i in range(len(df))]

    # Price numeric
    price_cols = ["price", "Celková cena:", "Celková cena"]
    df["price_czk"] = np.nan
    for i, row in df.iterrows():
        price_raw = ""
        for c in price_cols:
            if c in df.columns and pd.notna(row.get(c, None)):
                price_raw = str(row[c])
                break
        df.at[i, "price_czk"] = parse_price_czk(price_raw)

    # Areas
    for col in ["plocha_pozemku_m2", "uzitna_plocha_m2", "zastavena_plocha_m2", "celkova_plocha_m2"]:
        df[col] = np.nan
    for i, row in df.iterrows():
        areas = plochy_from_row(row)
        for k, v in areas.items():
            df.at[i, k] = v

    # ppsqm metrics
    def safe_div(a, b):
        try:
            a = float(a); b = float(b)
            return a / b if b not in (0, np.nan) and not pd.isna(b) else np.nan
        except:
            return np.nan

    df["ppsqm_uzitna"]    = [safe_div(a, b) for a, b in zip(df["price_czk"], df["uzitna_plocha_m2"])]
    df["ppsqm_pozemek"]   = [safe_div(a, b) for a, b in zip(df["price_czk"], df["plocha_pozemku_m2"])]
    df["ppsqm_zastavena"] = [safe_div(a, b) for a, b in zip(df["price_czk"], df["zastavena_plocha_m2"])]
    df["ppsqm_celkova"]   = [safe_div(a, b) for a, b in zip(df["price_czk"], df["celkova_plocha_m2"])]

    # Distance to work
    def to_float(x):
        try:
            return float(x)
        except:
            return np.nan
    lat = df.get("latitude", pd.Series([None]*len(df))).apply(to_float)
    lon = df.get("longitude", pd.Series([None]*len(df))).apply(to_float)
    df["distance_to_work_km"] = [
        haversine_km(la, lo, work_lat, work_lon) if not (pd.isna(la) or pd.isna(lo)) else np.nan
        for la, lo in zip(lat, lon)
    ]

    return df

# ---------- Sidebar: inputs ----------
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
    format_func=lambda x: x[1],
    index=0
)

# Weights for a simple combined score (lower is better)
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

# ---------- Derived / Filters ----------
ppsqm_col = area_choice[0]

# Normalize metrics for ranking (min-max), lower is better for both
def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return s
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        # All same; return 0.5 so it doesn't dominate
        return pd.Series([0.5 if not pd.isna(x) else np.nan for x in s], index=s.index)
    return (s - mn) / (mx - mn)

df["rank_metric_ppsqm"] = minmax(df[ppsqm_col])
df["rank_metric_dist"]  = minmax(df["distance_to_work_km"])
# Combined score: lower is better (weighted sum)
df["score"] = w_ppsqm * df["rank_metric_ppsqm"] + w_dist * df["rank_metric_dist"]

# Filters (create dynamic ranges from data)
def slider_range(series: pd.Series, label: str, step: float = 1.0):
    s = series.dropna().astype(float)
    if s.empty:
        return None
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

df_view = df[mask].copy()
df_view.sort_values("score", inplace=True)

# ---------- Header / KPIs ----------
st.title("🏠 House Hunter – Sreality analytics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Listings (shown / total)", f"{len(df_view)}/{len(df)}")
col2.metric("Median price", f"{int(np.nanmedian(df_view['price_czk'])):,} Kč".replace(",", " ") if not df_view['price_czk'].dropna().empty else "—")
col3.metric(f"Median {ppsqm_col}", f"{int(np.nanmedian(df_view[ppsqm_col])):,} Kč/m²".replace(",", " ") if not df_view[ppsqm_col].dropna().empty else "—")
col4.metric("Median distance", f"{np.nanmedian(df_view['distance_to_work_km']):.1f} km" if not df_view['distance_to_work_km'].dropna().empty else "—")

# ---------- Charts ----------
st.subheader("Price per m² ranking")
rank_df = df_view[["label", ppsqm_col, "score"]].dropna().sort_values(ppsqm_col, ascending=True)
if not rank_df.empty:
    chart = (
        alt.Chart(rank_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{ppsqm_col}:Q", title=f"{ppsqm_col} (CZK/m²)"),
            y=alt.Y("label:N", sort="-x", title=""),
            tooltip=["label", alt.Tooltip(f"{ppsqm_col}:Q", format=",.0f"), alt.Tooltip("score:Q", format=".2f")],
        )
        .properties(height=min(30 * len(rank_df), 900))
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No data for price-per-m² ranking.")

st.subheader("Distance vs. price per m²")
scatter_df = df_view[["distance_to_work_km", ppsqm_col, "label"]].dropna()
if not scatter_df.empty:
    scatter = (
        alt.Chart(scatter_df)
        .mark_point()
        .encode(
            x=alt.X("distance_to_work_km:Q", title="Distance to work (km)"),
            y=alt.Y(f"{ppsqm_col}:Q", title=f"{ppsqm_col} (CZK/m²)"),
            tooltip=["label", alt.Tooltip("distance_to_work_km:Q", format=".2f"), alt.Tooltip(f"{ppsqm_col}:Q", format=",.0f")],
        )
        .interactive()
        .properties(height=420)
    )
    st.altair_chart(scatter, use_container_width=True)
else:
    st.info("No data for scatter plot.")

st.subheader("Comparison heatmap (normalized)")
heat_cols = ["ppsqm_uzitna", "ppsqm_pozemek", "ppsqm_zastavena", "distance_to_work_km"]
avail_heat_cols = [c for c in heat_cols if c in df_view.columns and not df_view[c].dropna().empty]
if avail_heat_cols:
    # Build normalized matrix for selected rows
    H = df_view[["label"] + avail_heat_cols].dropna()
    if not H.empty:
        # Normalize each column
        for c in avail_heat_cols:
            H[c] = minmax(H[c])
        # Melt for Altair
        Hm = H.melt(id_vars="label", var_name="metric", value_name="normalized")
        heat = (
            alt.Chart(Hm)
            .mark_rect()
            .encode(
                y=alt.Y("label:N", sort="-x", title=""),
                x=alt.X("metric:N", title="Metric"),
                color=alt.Color("normalized:Q", title="normalized"),
                tooltip=["label", "metric", alt.Tooltip("normalized:Q", format=".2f")],
            )
            .properties(height=min(30 * len(H["label"].unique()), 900))
        )
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("No complete rows for heatmap.")
else:
    st.info("No metrics available for heatmap.")

# ---------- Map ----------
st.subheader("Map")
map_df = df_view[["label", "latitude", "longitude", "price_czk", ppsqm_col]].copy()
map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
map_df.dropna(subset=["latitude", "longitude"], inplace=True)

if not map_df.empty:
    # Simple color scaling on ppsqm (normalized 0..1)
    if map_df[ppsqm_col].notna().any():
        mm = minmax(map_df[ppsqm_col])
    else:
        mm = pd.Series([0.5] * len(map_df), index=map_df.index)
    map_df["norm"] = mm.fillna(0.5)

    # Color from norm (blue -> red) without specifying exact colors (pydeck default colormap isn't used,
    # so we create RGB ourselves based on norm)
    # 0 -> blue(80,120,200), 1 -> red(200,80,80)
    base1 = np.array([80,120,200])
    base2 = np.array([200,80,80])
    colors = (base1[None, :] * (1 - map_df["norm"].values[:, None]) + base2[None, :] * map_df["norm"].values[:, None]).astype(int)
    map_df["r"] = colors[:,0]; map_df["g"] = colors[:,1]; map_df["b"] = colors[:,2]

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
        "html": "<b>{label}</b><br/>Price: {price_czk}<br/>"+ppsqm_col+": {"+ppsqm_col+"}",
        "style": {"backgroundColor": "white", "color": "black"}
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
else:
    st.info("No geocoded listings to map.")

# ---------- Table & Download ----------
st.subheader("Table")
show_cols = [
    "label", "price_czk", "distance_to_work_km",
    "uzitna_plocha_m2", "plocha_pozemku_m2", "zastavena_plocha_m2", "celkova_plocha_m2",
    "ppsqm_uzitna", "ppsqm_pozemek", "ppsqm_zastavena", "ppsqm_celkova",
    "score", "url"
]
show_cols = [c for c in show_cols if c in df_view.columns]
st.dataframe(df_view[show_cols], use_container_width=True, hide_index=True)

st.markdown("#### Download enriched CSV")
dl_df = df_view.copy()
csv_bytes = dl_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=csv_bytes,
    file_name="listings_enriched_filtered.csv",
    mime="text/csv",
)

st.caption("Note: ‘score’ is a simple weighted sum of normalized price-per-m² and distance. Adjust the weight in the sidebar.")

# ---------- Footer ----------
st.markdown("---")
st.caption("Built for your Sreality scraper • Streamlit + Altair + PyDeck • tweak freely in app.py")
