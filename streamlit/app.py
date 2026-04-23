"""
Requirements (install in your environment):
- streamlit
- pandas
- plotly
- sqlalchemy
- psycopg2-binary
- python-dotenv
"""

from __future__ import annotations

import os
import re
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# -----------------------------------------------------------------------------
# Page / visual setup
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Chicago Movies in the Parks Dashboard")
USE_CSV_DEFAULT = os.getenv("USE_CSV", "1").strip().lower() not in {"0", "false", "no", "off"}

PALETTE = {
	"primary": "#1B4332",
	"secondary": "#2D6A4F",
	"accent": "#40916C",
	"warm": "#DDBEA9",
	"bg": "#F8F9FA",
	"muted": "#6C757D",
}

CHART_BG = "#0B1117"
CHART_GRID = "#2B3640"
CHART_TEXT = "#E8EEF2"

st.markdown(
	f"""
	<style>
	:root {{
		--primary-color: {PALETTE['secondary']};
	}}
	.stApp {{
		background: linear-gradient(180deg, {PALETTE['bg']} 0%, #EFF5F1 100%);
		--primary-color: {PALETTE['secondary']};
	}}
	.stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
		color: #1f2a2e;
	}}
	h1, h2, h3 {{
		color: {PALETTE['primary']};
	}}
	[data-testid="stSidebar"] {{
		background: #EEF4EF;
		border-right: 1px solid #D7E4DA;
	}}
	[data-testid="stSidebar"] * {{
		color: #1f2a2e;
	}}
	[data-testid="stSidebar"] [data-baseweb="tag"] {{
		background: {PALETTE['secondary']} !important;
		border: none !important;
		color: #ffffff !important;
	}}
	[data-testid="stSidebar"] [data-baseweb="tag"] * {{
		color: #ffffff !important;
	}}
	[data-baseweb="tag"] {{
		background: {PALETTE['secondary']} !important;
		border: none !important;
		color: #ffffff !important;
	}}
	[data-baseweb="tag"] * {{
		color: #ffffff !important;
	}}
	.stButton > button {{
		background: {PALETTE['primary']} !important;
		color: #ffffff !important;
		border: 1px solid {PALETTE['primary']} !important;
	}}
	.stButton > button:hover,
	.stButton > button:focus,
	.stButton > button:active {{
		background: {PALETTE['secondary']} !important;
		color: #ffffff !important;
		border: 1px solid {PALETTE['secondary']} !important;
	}}
	[data-testid="stSidebar"] .stButton > button,
	[data-testid="stSidebar"] .stButton > button * {{
		color: #ffffff !important;
	}}
	[data-baseweb="slider"] [role="slider"] {{
		background: {PALETTE['secondary']} !important;
		border-color: {PALETTE['secondary']} !important;
		box-shadow: none !important;
	}}
	[data-baseweb="slider"] [data-testid="stTickBarMax"] {{
		background: {PALETTE['secondary']} !important;
	}}
	[data-baseweb="slider"] [data-testid="stTickBarMin"] {{
		background: #BFD8C8 !important;
	}}
	.metric-card {{
		border: 1px solid #DDE5DF;
		border-radius: 12px;
		padding: 0.75rem 1rem;
		background: white;
	}}
	</style>
	""",
	unsafe_allow_html=True,
)

pio.templates["park_clean"] = go.layout.Template(
	layout=go.Layout(
		font={"color": CHART_TEXT, "size": 14},
		title={"font": {"color": CHART_TEXT, "size": 22}},
		paper_bgcolor=CHART_BG,
		plot_bgcolor=CHART_BG,
		xaxis={
			"title": {"font": {"color": CHART_TEXT}},
			"tickfont": {"color": CHART_TEXT},
			"gridcolor": CHART_GRID,
			"zerolinecolor": CHART_GRID,
		},
		yaxis={
			"title": {"font": {"color": CHART_TEXT}},
			"tickfont": {"color": CHART_TEXT},
			"gridcolor": CHART_GRID,
			"zerolinecolor": CHART_GRID,
		},
		legend={"font": {"color": CHART_TEXT}},
	)
)
pio.templates.default = "plotly_white+park_clean"


def _style_chart(fig: go.Figure, **layout_kwargs) -> None:
	fig.update_layout(
		plot_bgcolor=CHART_BG,
		paper_bgcolor=CHART_BG,
		font={"color": CHART_TEXT},
		legend={"font": {"color": CHART_TEXT}},
		**layout_kwargs,
	)
	fig.update_xaxes(
		gridcolor=CHART_GRID,
		zerolinecolor=CHART_GRID,
		tickfont={"color": CHART_TEXT},
		title_font={"color": CHART_TEXT},
	)
	fig.update_yaxes(
		gridcolor=CHART_GRID,
		zerolinecolor=CHART_GRID,
		tickfont={"color": CHART_TEXT},
		title_font={"color": CHART_TEXT},
	)


@st.cache_data(ttl=604800, show_spinner=False)
def _fetch_zip_centroid(zip_code: str) -> tuple[float, float] | None:
	base_url = "https://nominatim.openstreetmap.org/search"
	headers = {"User-Agent": "DAEN328-Streamlit-Dashboard/1.0"}

	param_sets = [
		{
			"format": "json",
			"countrycodes": "us",
			"postalcode": zip_code,
			"city": "Chicago",
			"state": "Illinois",
			"limit": 1,
		},
		{
			"format": "json",
			"q": f"{zip_code}, Chicago, Illinois",
			"limit": 1,
		},
	]

	for params in param_sets:
		url = f"{base_url}?{urlencode(params)}"
		try:
			request = Request(url, headers=headers)
			with urlopen(request, timeout=5) as response:
				payload = response.read().decode("utf-8")
				results = json.loads(payload)
				if isinstance(results, list) and len(results) > 0:
					lat = float(results[0]["lat"])
					lon = float(results[0]["lon"])
					return lat, lon
		except Exception:
			continue

	return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_zip_centroids(zip_codes: tuple[str, ...]) -> pd.DataFrame:
	records = []
	for zip_code in zip_codes:
		coords = _fetch_zip_centroid(str(zip_code))
		if coords is None:
			continue
		records.append({"zip_code": str(zip_code), "lat": coords[0], "lon": coords[1]})

	return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Environment / database helpers
# -----------------------------------------------------------------------------
def _load_env() -> None:
	here = Path(__file__).resolve()
	project_root = here.parent.parent
	env_candidates = [project_root / ".env", Path.cwd() / ".env"]
	for env_path in env_candidates:
		if env_path.exists():
			load_dotenv(env_path)


def _db_url() -> str:
	host = os.getenv("DB_HOST", "localhost")
	port = os.getenv("DB_PORT", "5432")
	name = os.getenv("DB_NAME", "movies_db")
	user = os.getenv("DB_USER", "postgres")
	password = os.getenv("DB_PASSWORD", "")
	return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def _qi(identifier: str) -> str:
	return '"' + identifier.replace('"', '""') + '"'


def _qtable(schema_name: str, table_name: str) -> str:
	return f"{_qi(schema_name)}.{_qi(table_name)}"


def _project_root() -> Path:
	return Path(__file__).resolve().parent.parent


def _data_dir() -> Path:
	return Path(os.getenv("DATA_DIR", str(_project_root() / "data")))


def _csv_paths() -> tuple[Path, Path]:
	data_dir = _data_dir()
	return data_dir / "cleaned_movies_final.csv", data_dir / "census_data.csv"


def _safe_zip_series(value: pd.Series | pd.DataFrame) -> pd.Series:
	if isinstance(value, pd.DataFrame):
		value = value.iloc[:, 0]
	return pd.Series(value)


def _normalize_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
	df = df.loc[:, ~df.columns.duplicated()].copy()

	if "event_date" not in df.columns:
		if "Date" in df.columns:
			df["event_date"] = pd.to_datetime(df["Date"], errors="coerce")
		elif "date" in df.columns:
			df["event_date"] = pd.to_datetime(df["date"], errors="coerce")
		else:
			df["event_date"] = pd.NaT
	else:
		df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

	df["year"] = df["event_date"].dt.year
	df["month"] = df["event_date"].dt.month
	df["month_name"] = df["event_date"].dt.month_name()

	if "rating_raw" not in df.columns:
		for candidate in ["Rating", "rating"]:
			if candidate in df.columns:
				df["rating_raw"] = df[candidate]
				break
		else:
			df["rating_raw"] = pd.NA

	if "movie_name" not in df.columns:
		for candidate in ["Movie Name", "movie_name", "moviename"]:
			if candidate in df.columns:
				df["movie_name"] = df[candidate]
				break
		else:
			df["movie_name"] = pd.NA

	if "park_name" not in df.columns:
		for candidate in ["Park Name", "park_name", "park"]:
			if candidate in df.columns:
				df["park_name"] = df[candidate]
				break
		else:
			df["park_name"] = pd.NA

	if "zip_code" not in df.columns:
		for candidate in ["Zip", "zip", "zip_code"]:
			if candidate in df.columns:
				df["zip_code"] = df[candidate]
				break
		else:
			df["zip_code"] = "Unknown"

	zip_series = _safe_zip_series(df["zip_code"])
	df["zip_code"] = (
		zip_series.astype(str)
		.str.extract(r"(\d{5})", expand=False)
		.fillna("Unknown")
	)

	if "location_group" not in df.columns:
		df["location_group"] = df["zip_code"]

	if "rating_num" not in df.columns:
		df["rating_num"] = _rating_to_numeric(pd.Series(df["rating_raw"]))

	if "income" not in df.columns:
		income_col = _pick_col(df.columns.tolist(), ["income", "median_household_income", "b19013"])
		df["income"] = pd.to_numeric(df[income_col], errors="coerce") if income_col else np.nan

	if "median_age" not in df.columns:
		age_col = _pick_col(df.columns.tolist(), ["median_age", "b01002", "age"])
		df["median_age"] = pd.to_numeric(df[age_col], errors="coerce") if age_col else np.nan

	if "edu_rate" not in df.columns:
		df["edu_rate"] = _extract_bachelor_plus(df, {})

	df["location_group"] = (
		df["location_group"]
		.fillna(df["zip_code"])
		.replace({"Unknown": np.nan})
		.fillna(df["park_name"])
		.fillna("Unknown")
	)

	return df


@st.cache_data(ttl=300, show_spinner=False)
def get_csv_dataset(movies_path: str, census_path: str) -> pd.DataFrame:
	movies = pd.read_csv(movies_path)
	census = pd.read_csv(census_path)

	movies = movies.rename(
		columns={
			"Movie Name": "movie_name",
			"Address": "address",
			"Rating": "rating_raw",
			"Year": "year",
			"Date": "event_date",
			"Closed Captioning": "closed_captioning",
			"Zip": "zip_code",
			"Park Name": "park_name",
		}
	)

	census = census.rename(
		columns={
			"Zip": "zip_code",
			"Median Age": "median_age",
			"Median Household Income": "income",
			"White Alone Population": "white_alone_population",
			"Black Alone Population": "black_alone_population",
			"Asian Alone Population": "asian_alone_population",
			"Other Race Alone Population": "other_race_alone_population",
			"Marital Status Universe": "marital_status_universe",
			"Married Male Population": "married_male_population",
			"Married Female Population": "married_female_population",
			"Women With Birth Last 12 Months": "women_with_birth_last_12_months",
			"Education Population Total": "education_population_total",
			"Bachelors Degree Population": "bachelors_degree_population",
			"Masters Degree Population": "masters_degree_population",
			"Professional School Degree Population": "professional_school_degree_population",
			"Doctorate Degree Population": "doctorate_degree_population",
			"Total Population": "total_population",
		}
	)

	for col in ["zip_code", "income", "median_age", "white_alone_population", "black_alone_population", "asian_alone_population", "other_race_alone_population", "education_population_total", "bachelors_degree_population", "masters_degree_population", "professional_school_degree_population", "doctorate_degree_population"]:
		if col in census.columns:
			census[col] = pd.to_numeric(census[col], errors="coerce")

	if "zip_code" in movies.columns:
		movies["zip_code"] = movies["zip_code"].astype(str).str.extract(r"(\d{5})", expand=False).fillna("Unknown")

	merged = movies.merge(census, on="zip_code", how="left")
	merged["predominant_race"] = merged.apply(_calc_predominant_race, axis=1)
	return _normalize_analysis_frame(merged)


def _calc_predominant_race(row: pd.Series) -> str | None:
	races = {
		"White": pd.to_numeric(row.get("white_alone_population"), errors="coerce"),
		"Black": pd.to_numeric(row.get("black_alone_population"), errors="coerce"),
		"Asian": pd.to_numeric(row.get("asian_alone_population"), errors="coerce"),
		"Other": pd.to_numeric(row.get("other_race_alone_population"), errors="coerce"),
	}
	races = {key: value for key, value in races.items() if pd.notna(value)}
	if not races:
		return None
	return max(races, key=races.get)


@st.cache_data(ttl=180, show_spinner=False)
def get_csv_refresh(movies_path: str, census_path: str) -> str:
	paths = [Path(movies_path), Path(census_path)]
	mtimes = [path.stat().st_mtime for path in paths if path.exists()]
	if not mtimes:
		return "CSV files not found"
	return datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M:%S") + " (CSV snapshot)"

@st.cache_data(ttl=300, show_spinner=False)
def get_schema_snapshot(db_url: str) -> pd.DataFrame:
	sql = """
	SELECT
		table_schema,
		table_name,
		column_name,
		data_type
	FROM information_schema.columns
	WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
	ORDER BY table_schema, table_name, ordinal_position;
	"""
	engine = create_engine(db_url)
	with engine.connect() as conn:
		return pd.read_sql(text(sql), conn)


def _norm(value: str) -> str:
	return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _pick_col(columns: list[str], patterns: list[str]) -> str | None:
	norm_to_raw = {_norm(c): c for c in columns}
	for pattern in patterns:
		for ncol, raw in norm_to_raw.items():
			if pattern in ncol:
				return raw
	return None


def _score_events_table(columns: list[str]) -> int:
	ncols = [_norm(c) for c in columns]
	score = 0
	if any("date" in c for c in ncols):
		score += 4
	if any("rating" in c for c in ncols):
		score += 3
	if any("park" in c for c in ncols):
		score += 2
	if any("movie" in c or "title" in c for c in ncols):
		score += 1
	if any("event" in c for c in ncols):
		score += 1
	return score


def _score_parks_table(columns: list[str], table_name: str) -> int:
	ncols = [_norm(c) for c in columns]
	score = 0
	if "park" in _norm(table_name):
		score += 2
	if any("park_name" in c or c == "park" or c == "name" for c in ncols):
		score += 4
	if any("zip" in c for c in ncols):
		score += 3
	if any(c.endswith("id") for c in ncols):
		score += 1
	return score


def _score_census_table(columns: list[str], table_name: str) -> int:
	ncols = [_norm(c) for c in columns]
	score = 0
	if "zip" in _norm(table_name) or "census" in _norm(table_name):
		score += 2
	if any("zip" in c for c in ncols):
		score += 4
	if any("income" in c for c in ncols):
		score += 3
	if any("age" in c for c in ncols):
		score += 2
	if any("edu" in c or "bachelor" in c for c in ncols):
		score += 2
	if any("race" in c or "white" in c or "black" in c or "asian" in c for c in ncols):
		score += 1
	return score


def discover_model(schema_df: pd.DataFrame) -> dict:
	grouped = (
		schema_df.groupby(["table_schema", "table_name"], as_index=False)
		.agg(columns=("column_name", list))
	)
	if grouped.empty:
		return {}

	rows = grouped.to_dict("records")

	events_row = max(rows, key=lambda r: _score_events_table(r["columns"]))
	parks_row = max(rows, key=lambda r: _score_parks_table(r["columns"], r["table_name"]))
	census_row = max(rows, key=lambda r: _score_census_table(r["columns"], r["table_name"]))

	events_cols = events_row["columns"]
	parks_cols = parks_row["columns"]
	census_cols = census_row["columns"]

	events_date_col = _pick_col(events_cols, ["date", "event_date", "start_date"]) or events_cols[0]
	events_rating_col = _pick_col(events_cols, ["rating", "score", "movie_rating"])
	events_park_fk_col = _pick_col(events_cols, ["park_id", "venue_id", "location_id", "park"])
	events_zip_col = _pick_col(events_cols, ["zip", "postal"])
	events_movie_col = _pick_col(events_cols, ["movie", "title", "film"])

	parks_id_col = _pick_col(parks_cols, ["park_id", "id"])
	parks_name_col = _pick_col(parks_cols, ["park_name", "park", "name", "venue"])
	parks_zip_col = _pick_col(parks_cols, ["zip", "postal"])

	census_zip_col = _pick_col(census_cols, ["zip", "postal"])
	census_income_col = _pick_col(census_cols, ["income", "median_household_income", "b19013"])
	census_age_col = _pick_col(census_cols, ["median_age", "age", "b01002"])
	census_edu_rate_col = _pick_col(census_cols, ["edu_rate", "education_rate", "bachelor"])

	census_race_cols = [
		c
		for c in census_cols
		if any(tag in _norm(c) for tag in ["white", "black", "asian", "other", "hispanic", "race"])
	]

	return {
		"events": {
			"schema": events_row["table_schema"],
			"table": events_row["table_name"],
			"columns": events_cols,
			"date_col": events_date_col,
			"rating_col": events_rating_col,
			"park_fk_col": events_park_fk_col,
			"zip_col": events_zip_col,
			"movie_col": events_movie_col,
		},
		"parks": {
			"schema": parks_row["table_schema"],
			"table": parks_row["table_name"],
			"columns": parks_cols,
			"id_col": parks_id_col,
			"name_col": parks_name_col,
			"zip_col": parks_zip_col,
		},
		"census": {
			"schema": census_row["table_schema"],
			"table": census_row["table_name"],
			"columns": census_cols,
			"zip_col": census_zip_col,
			"income_col": census_income_col,
			"age_col": census_age_col,
			"edu_rate_col": census_edu_rate_col,
			"race_cols": census_race_cols,
		},
	}


def _extract_bachelor_plus(df: pd.DataFrame, census_cols: dict) -> pd.Series:
	"""
	Derives Bachelor's+ attainment rate (%).
	Priority:
	1) use existing edu_rate-style column when present
	2) compute (bachelors + masters + professional + doctorate) / education_total * 100
	"""
	def find_col(patterns: list[str]) -> str | None:
		norm_to_raw = {_norm(c): c for c in df.columns}
		for pattern in patterns:
			for ncol, raw in norm_to_raw.items():
				if pattern in ncol:
					return raw
		return None

	edu_col = find_col(["edu_rate", "education_rate"])
	if edu_col:
		return pd.to_numeric(df[edu_col], errors="coerce")

	bach = find_col(["bachelors_degree_population", "b15003_022e", "bachelor"])
	mast = find_col(["masters_degree_population", "b15003_023e", "master"])
	prof = find_col(["professional_school_degree_population", "b15003_024e", "professional"])
	doc = find_col(["doctorate_degree_population", "b15003_025e", "doctorate"])
	total = find_col(["education_population_total", "b15003_001e", "education_total"])

	if all([bach, mast, prof, doc, total]):
		numerator = (
			pd.to_numeric(df[bach], errors="coerce").fillna(0)
			+ pd.to_numeric(df[mast], errors="coerce").fillna(0)
			+ pd.to_numeric(df[prof], errors="coerce").fillna(0)
			+ pd.to_numeric(df[doc], errors="coerce").fillna(0)
		)
		denom = pd.to_numeric(df[total], errors="coerce").replace(0, np.nan)
		return (numerator / denom) * 100

	return pd.Series(np.nan, index=df.index)


def _rating_to_numeric(series: pd.Series) -> pd.Series:
	mapping = {
		"G": 1,
		"PG": 2,
		"PG13": 3,
		"PG_13": 3,
		"PG-13": 3,
		"R": 4,
		"NC17": 5,
		"NC_17": 5,
		"NC-17": 5,
		"NR": np.nan,
		"UR": np.nan,
	}
	num = pd.to_numeric(series, errors="coerce")
	text_series = series.astype(str).str.upper().str.replace(" ", "", regex=False)
	mapped = text_series.map(mapping)
	return num.fillna(mapped)


@st.cache_data(ttl=300, show_spinner=False)
def get_base_dataset(db_url: str, model: dict) -> pd.DataFrame:
	events = model["events"]
	parks = model["parks"]
	census = model["census"]

	e_alias = "e"
	p_alias = "p"
	c_alias = "c"

	e_tbl = _qtable(events["schema"], events["table"])
	p_tbl = _qtable(parks["schema"], parks["table"])
	c_tbl = _qtable(census["schema"], census["table"])

	select_parts = [
		f"{e_alias}.{_qi(events['date_col'])}::date AS event_date",
	]

	if events.get("rating_col"):
		select_parts.append(f"{e_alias}.{_qi(events['rating_col'])} AS rating_raw")
	else:
		select_parts.append("NULL::text AS rating_raw")

	if events.get("movie_col"):
		select_parts.append(f"{e_alias}.{_qi(events['movie_col'])} AS movie_name")
	else:
		select_parts.append("NULL::text AS movie_name")

	park_name_expr = "NULL::text"
	if parks.get("name_col"):
		park_name_expr = f"{p_alias}.{_qi(parks['name_col'])}"
	elif events.get("park_fk_col"):
		park_name_expr = f"{e_alias}.{_qi(events['park_fk_col'])}::text"
	select_parts.append(f"{park_name_expr} AS park_name")

	event_zip_expr = (
		f"{e_alias}.{_qi(events['zip_col'])}::text"
		if events.get("zip_col")
		else "NULL::text"
	)
	park_zip_expr = (
		f"{p_alias}.{_qi(parks['zip_col'])}::text"
		if parks.get("zip_col")
		else "NULL::text"
	)
	select_parts.append(f"COALESCE({park_zip_expr}, {event_zip_expr}) AS zip_code")

	neighborhood_like_col = _pick_col(
		parks.get("columns", []) + events.get("columns", []) + census.get("columns", []),
		["neighborhood", "district", "community", "region", "area", "city"],
	)
	if neighborhood_like_col:
		if neighborhood_like_col in parks.get("columns", []):
			select_parts.append(f"{p_alias}.{_qi(neighborhood_like_col)}::text AS location_group")
		elif neighborhood_like_col in events.get("columns", []):
			select_parts.append(f"{e_alias}.{_qi(neighborhood_like_col)}::text AS location_group")
		else:
			select_parts.append(f"{c_alias}.{_qi(neighborhood_like_col)}::text AS location_group")
	else:
		select_parts.append("NULL::text AS location_group")

	used_aliases = {"event_date", "rating_raw", "movie_name", "park_name", "zip_code", "location_group"}
	for col_name in census.get("columns", []):
		base_alias = _norm(col_name)
		if not base_alias:
			continue

		alias = base_alias
		if alias in used_aliases:
			alias = f"c_{alias}"

		suffix = 2
		while alias in used_aliases:
			alias = f"{base_alias}_{suffix}"
			suffix += 1

		used_aliases.add(alias)
		select_parts.append(f"{c_alias}.{_qi(col_name)} AS {alias}")

	joins = []
	from_clause = f"FROM {e_tbl} {e_alias}"

	can_join_parks = all([events.get("park_fk_col"), parks.get("id_col")])
	if can_join_parks:
		joins.append(
			f"LEFT JOIN {p_tbl} {p_alias} "
			f"ON {e_alias}.{_qi(events['park_fk_col'])} = {p_alias}.{_qi(parks['id_col'])}"
		)

	can_join_census = census.get("zip_col") is not None
	if can_join_census:
		left_zip = park_zip_expr if parks.get("zip_col") else event_zip_expr
		joins.append(
			f"LEFT JOIN {c_tbl} {c_alias} "
			f"ON {c_alias}.{_qi(census['zip_col'])}::text = {left_zip}"
		)

	sql = "\n".join([
		"SELECT",
		"    " + ",\n    ".join(select_parts),
		from_clause,
		*joins,
	])

	engine = create_engine(db_url)
	with engine.connect() as conn:
		df = pd.read_sql(text(sql), conn)

	# If any duplicate column names still slip through, keep first occurrence only.
	df = df.loc[:, ~df.columns.duplicated()]

	df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
	df["year"] = df["event_date"].dt.year
	df["month"] = df["event_date"].dt.month
	df["month_name"] = df["event_date"].dt.month_name()
	zip_series = df["zip_code"]
	if isinstance(zip_series, pd.DataFrame):
		zip_series = zip_series.iloc[:, 0]

	df["zip_code"] = (
		zip_series
		.astype(str)
		.str.extract(r"(\d{5})", expand=False)
		.fillna("Unknown")
	)

	df["rating_num"] = _rating_to_numeric(df["rating_raw"])

	if "location_group" not in df.columns:
		df["location_group"] = np.nan

	df["location_group"] = (
		df["location_group"]
		.fillna(df["zip_code"])
		.replace({"Unknown": np.nan})
		.fillna(df["park_name"])
		.fillna("Unknown")
	)

	if "income" in df.columns:
		df["income"] = pd.to_numeric(df["income"], errors="coerce")
	else:
		income_col = _pick_col(df.columns.tolist(), ["income", "median_household_income", "b19013"])
		df["income"] = pd.to_numeric(df[income_col], errors="coerce") if income_col else np.nan

	if "median_age" in df.columns:
		df["median_age"] = pd.to_numeric(df["median_age"], errors="coerce")
	else:
		age_col = _pick_col(df.columns.tolist(), ["median_age", "b01002", "age"])
		df["median_age"] = pd.to_numeric(df[age_col], errors="coerce") if age_col else np.nan

	df["edu_rate"] = _extract_bachelor_plus(df, census)
	return df


@st.cache_data(ttl=180, show_spinner=False)
def get_last_refresh(db_url: str, model: dict) -> str:
	ts_candidates = ["updated_at", "created_at", "loaded_at", "etl_loaded_at", "inserted_at"]
	all_tables = [model["events"], model["parks"], model["census"]]

	engine = create_engine(db_url)
	with engine.connect() as conn:
		for table_info in all_tables:
			cols = table_info.get("columns", [])
			ts_col = _pick_col(cols, ts_candidates)
			if not ts_col:
				continue

			sql = (
				f"SELECT MAX({_qi(ts_col)}) AS max_ts "
				f"FROM {_qtable(table_info['schema'], table_info['table'])}"
			)
			out = pd.read_sql(text(sql), conn)
			if not out.empty and pd.notna(out.loc[0, "max_ts"]):
				return pd.to_datetime(out.loc[0, "max_ts"]).strftime("%Y-%m-%d %H:%M:%S")

	return f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local dashboard read time)"


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
	filtered = df.copy()

	valid_years = sorted([int(y) for y in filtered["year"].dropna().unique()])
	if valid_years:
		default_years = (min(valid_years), max(valid_years))
	else:
		default_years = (2014, 2019)

	st.sidebar.header("Filters")
	if st.sidebar.button("Reset Filters"):
		for key in ["year_range", "months", "zips", "parks", "rating_range"]:
			if key in st.session_state:
				del st.session_state[key]
		st.rerun()

	year_range = st.sidebar.slider(
		"Year range",
		min_value=default_years[0],
		max_value=default_years[1],
		value=default_years,
		key="year_range",
	)

	month_options = [
		"January", "February", "March", "April", "May", "June",
		"July", "August", "September", "October", "November", "December",
	]
	present_months = [m for m in month_options if m in set(filtered["month_name"].dropna())]
	if "months" not in st.session_state:
		st.session_state["months"] = present_months

	month_action_col1, month_action_col2 = st.sidebar.columns(2)
	with month_action_col1:
		if st.button("Select All Months"):
			st.session_state["months"] = month_options.copy()
	with month_action_col2:
		if st.button("Clear Months"):
			st.session_state["months"] = []

	selected_months = st.sidebar.multiselect(
		"Month(s)",
		options=month_options,
		key="months",
	)

	zip_options = sorted([z for z in filtered["zip_code"].dropna().unique() if z != "Unknown"])
	if "zips" not in st.session_state:
		st.session_state["zips"] = []

	zip_action_col1, zip_action_col2 = st.sidebar.columns(2)
	with zip_action_col1:
		if st.button("Select All ZIPs"):
			st.session_state["zips"] = zip_options.copy()
	with zip_action_col2:
		if st.button("Clear ZIPs"):
			st.session_state["zips"] = []

	selected_zips = st.sidebar.multiselect(
		"ZIP(s)",
		options=zip_options,
		key="zips",
	)

	park_options = sorted([p for p in filtered["park_name"].dropna().unique() if str(p).strip()])
	selected_parks = st.sidebar.multiselect(
		"Park(s)",
		options=park_options,
		default=[],
		key="parks",
	)

	valid_ratings = filtered["rating_num"].dropna()
	if not valid_ratings.empty:
		r_min = float(valid_ratings.min())
		r_max = float(valid_ratings.max())
		rating_range = st.sidebar.slider(
			"Rating range",
			min_value=float(np.floor(r_min)),
			max_value=float(np.ceil(r_max)),
			value=(float(np.floor(r_min)), float(np.ceil(r_max))),
			step=0.5,
			key="rating_range",
		)
	else:
		rating_range = None

	filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

	if selected_months:
		filtered = filtered[filtered["month_name"].isin(selected_months)]
	if selected_zips:
		filtered = filtered[filtered["zip_code"].isin(selected_zips)]
	if selected_parks:
		filtered = filtered[filtered["park_name"].isin(selected_parks)]
	if rating_range is not None:
		filtered = filtered[
			filtered["rating_num"].isna()
			| ((filtered["rating_num"] >= rating_range[0]) & (filtered["rating_num"] <= rating_range[1]))
		]

	return filtered


def add_trendline(fig: go.Figure, x: pd.Series, y: pd.Series, color: str = PALETTE["secondary"]) -> None:
	clean = pd.DataFrame({"x": x, "y": y}).dropna()
	if len(clean) < 2:
		return
	slope, intercept = np.polyfit(clean["x"], clean["y"], 1)
	x_line = np.linspace(clean["x"].min(), clean["x"].max(), 100)
	y_line = slope * x_line + intercept
	fig.add_trace(
		go.Scatter(
			x=x_line,
			y=y_line,
			mode="lines",
			name="Linear trend",
			line={"color": color, "width": 2, "dash": "dash"},
		)
	)


def main() -> None:
	_load_env()
	db_url = _db_url()

	st.title("Chicago Movies in the Parks: Insights Dashboard")
	st.write(
		"Interactive analysis of movie events and demographics from the local Postgres ETL output. "
		"All table and column mappings are discovered at runtime from the database schema."
	)

	try:
		schema_df = get_schema_snapshot(db_url)
		model = discover_model(schema_df)
		if not model:
			st.error("No tables were discovered in the connected database.")
			return

		data = get_base_dataset(db_url, model)
		if data.empty:
			st.warning("Connected successfully, but no event records were returned.")
			return

		refreshed_at = get_last_refresh(db_url, model)
		st.caption(f"Data last refreshed: {refreshed_at}. Source: local Postgres database.")

	except Exception as exc:
		st.error("Unable to connect or query local Postgres. Check DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD in .env.")
		st.exception(exc)
		return

	filtered = apply_filters(data)
	if filtered.empty:
		st.warning("No rows match the current filters. Please broaden your filter selection.")
		return

	tabs = st.tabs(
		[
			"Rating by Location",
			"Location Distribution",
			"Park Consistency",
			"Seasonality",
			"Income vs Quality",
			"Education vs Concentration",
			"Age vs Season Timing",
			"Race & Park Usage",
			"Event Density Map",
		]
	)

	# Q1
	with tabs[0]:
		st.subheader("Do certain ZIP codes/neighborhoods get higher movie ratings?")
		st.write("Average rating is ranked by location, with event count available on hover to keep comparisons grounded in sample size.")
		st.caption("Rating conversion used for analysis: G=1, PG=2, PG-13=3, R=4, NC-17=5 (NR/UR excluded from numeric averages).")

		q1 = filtered.dropna(subset=["rating_num"]).groupby("zip_code", as_index=False).agg(
			avg_rating=("rating_num", "mean"),
			event_count=("rating_num", "size"),
		)
		q1 = q1[q1["zip_code"] != "Unknown"].sort_values("avg_rating", ascending=False)

		top_n = st.slider("Top N ZIPs", min_value=5, max_value=30, value=10, step=1)
		q1_top = q1.head(top_n)

		if q1_top.empty:
			st.warning("No rating data available for the current filters.")
		else:
			fig_q1 = px.bar(
				q1_top,
				x="zip_code",
				y="avg_rating",
				color_discrete_sequence=[PALETTE["accent"]],
				hover_data={"event_count": True, "avg_rating": ":.2f"},
				title="Average Rating by ZIP (Ranked)",
			)
			_style_chart(fig_q1)
			st.plotly_chart(fig_q1, use_container_width=True)

			if len(q1_top) <= 10:
				box_df = filtered[filtered["zip_code"].isin(q1_top["zip_code"])].dropna(subset=["rating_num"])
				if not box_df.empty:
					fig_box = px.box(
						box_df,
						x="zip_code",
						y="rating_num",
						color_discrete_sequence=[PALETTE["secondary"]],
						title="Rating Distribution by ZIP (Top N)",
					)
					_style_chart(fig_box)
					st.plotly_chart(fig_box, use_container_width=True)

	# Q2
	with tabs[1]:
		st.subheader("What neighborhoods/areas host the most events?")
		st.write("This ranked view highlights where programming is concentrated geographically using the best available location grouping.")

		q2 = (
			filtered.groupby("location_group", as_index=False)
			.agg(event_count=("event_date", "size"))
			.sort_values("event_count", ascending=False)
			.head(20)
		)
		if q2.empty:
			st.warning("No location data is available for the current filters.")
		else:
			fig_q2 = px.bar(
				q2.sort_values("event_count"),
				x="event_count",
				y="location_group",
				orientation="h",
				color_discrete_sequence=[PALETTE["secondary"]],
				title="Top Locations by Event Count",
			)
			_style_chart(fig_q2, yaxis_title="Location group")
			st.plotly_chart(fig_q2, use_container_width=True)

	# Q3
	with tabs[2]:
		st.subheader("Which top 5 parks are the most consistent venues across all six years?")
		st.write("Consistency score is defined as distinct active years divided by years in range, then filtered to parks with enough events.")

		year_count = max(1, int(filtered["year"].nunique()))
		min_events = st.slider("Minimum events required", 1, 30, 5)

		park_year = filtered.groupby(["park_name", "year"], as_index=False).size().rename(columns={"size": "events"})
		park_stats = filtered.groupby("park_name", as_index=False).agg(
			total_events=("event_date", "size"),
			active_years=("year", "nunique"),
		)
		park_stats["consistency_score"] = park_stats["active_years"] / year_count
		park_stats = park_stats[park_stats["total_events"] >= min_events]
		park_top5 = park_stats.sort_values(["consistency_score", "total_events"], ascending=[False, False]).head(5)

		if park_top5.empty:
			st.warning("No parks meet the minimum event threshold for the selected filters.")
		else:
			fig_q3 = px.bar(
				park_top5.sort_values("consistency_score"),
				x="consistency_score",
				y="park_name",
				orientation="h",
				color="total_events",
				color_continuous_scale=[[0, PALETTE["warm"]], [1, PALETTE["accent"]]],
				title="Top 5 Park Consistency Scores",
			)
			_style_chart(fig_q3)
			st.plotly_chart(fig_q3, use_container_width=True)

			heat = park_year[park_year["park_name"].isin(park_top5["park_name"])].copy()
			heat_map = heat.pivot_table(index="park_name", columns="year", values="events", fill_value=0)
			fig_heat = px.imshow(
				heat_map,
				aspect="auto",
				color_continuous_scale=[[0, "#E9F2EC"], [1, PALETTE["primary"]]],
				title="Year-by-Year Presence Heatmap (Counts)",
			)
			st.plotly_chart(fig_heat, use_container_width=True)

	# Q4
	with tabs[3]:
		st.subheader("When does the Movie Season peak?")
		st.write("Monthly event totals reveal the seasonal demand profile, and the KPI identifies the strongest month in the filtered period.")

		month_order = {i: m for i, m in enumerate([
			"January", "February", "March", "April", "May", "June",
			"July", "August", "September", "October", "November", "December",
		], start=1)}

		by_month = filtered.groupby("month", as_index=False).agg(events=("event_date", "size"))
		by_month["month_name"] = by_month["month"].map(month_order)
		by_month = by_month.sort_values("month")

		if by_month.empty:
			st.warning("No monthly date information is available for this filter set.")
		else:
			peak_row = by_month.loc[by_month["events"].idxmax()]
			st.metric("Peak month", f"{peak_row['month_name']}", f"{int(peak_row['events'])} events")

			fig_q4 = px.line(
				by_month,
				x="month_name",
				y="events",
				markers=True,
				color_discrete_sequence=[PALETTE["accent"]],
				title="Events by Month (Aggregated Across Years)",
			)
			_style_chart(fig_q4)
			st.plotly_chart(fig_q4, use_container_width=True)

			by_month_year = filtered.groupby(["year", "month"], as_index=False).agg(events=("event_date", "size"))
			by_month_year["month_name"] = by_month_year["month"].map(month_order)
			by_month_year = by_month_year.sort_values(["year", "month"])
			fig_q4b = px.line(
				by_month_year,
				x="month_name",
				y="events",
				color="year",
				markers=True,
				title="Monthly Seasonality by Year",
				color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["warm"], "#7A9E7E", "#5A7D6B"],
			)
			_style_chart(fig_q4b)
			st.plotly_chart(fig_q4b, use_container_width=True)

	# Q5
	with tabs[4]:
		st.subheader("Do higher-income ZIP codes tend to have higher-rated movies?")
		st.write("Each bubble represents a ZIP code, linking median income to average rating with bubble size reflecting event volume.")

		q5 = filtered.groupby("zip_code", as_index=False).agg(
			avg_rating=("rating_num", "mean"),
			event_count=("event_date", "size"),
			income=("income", "mean"),
		)
		q5 = q5.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_rating", "income"])
		q5 = q5[q5["zip_code"] != "Unknown"]

		if q5.empty:
			st.warning("Income and rating overlap is unavailable for the selected filters.")
		else:
			corr = q5["income"].corr(q5["avg_rating"])
			st.caption(f"Correlation (income vs average rating): {corr:.3f}" if pd.notna(corr) else "Correlation unavailable")

			fig_q5 = px.scatter(
				q5,
				x="income",
				y="avg_rating",
				size="event_count",
				hover_name="zip_code",
				color_discrete_sequence=[PALETTE["accent"]],
				title="Income vs Average Movie Rating by ZIP",
			)
			add_trendline(fig_q5, q5["income"], q5["avg_rating"])
			_style_chart(fig_q5)
			st.plotly_chart(fig_q5, use_container_width=True)

	# Q6
	with tabs[5]:
		st.subheader("Are ZIP codes with higher educational attainment hosting more events per year?")
		st.write("Educational attainment is measured as Bachelor's+ share when available, compared against events per year in the selected range.")

		n_years = max(1, filtered["year"].nunique())
		q6 = filtered.groupby("zip_code", as_index=False).agg(
			edu_rate=("edu_rate", "mean"),
			event_count=("event_date", "size"),
		)
		q6["events_per_year"] = q6["event_count"] / n_years
		q6 = q6.replace([np.inf, -np.inf], np.nan).dropna(subset=["edu_rate", "events_per_year"])
		q6 = q6[q6["zip_code"] != "Unknown"]

		if q6.empty:
			st.warning("Education metrics are not available in the currently discovered schema.")
		else:
			fig_q6 = px.scatter(
				q6,
				x="edu_rate",
				y="events_per_year",
				hover_name="zip_code",
				size="event_count",
				color_discrete_sequence=[PALETTE["secondary"]],
				title="Bachelor's+ Rate vs Events per Year",
			)
			add_trendline(fig_q6, q6["edu_rate"], q6["events_per_year"], color=PALETTE["primary"])
			_style_chart(fig_q6, xaxis_title="Bachelor's+ rate (%)")
			st.plotly_chart(fig_q6, use_container_width=True)

	# Q7
	with tabs[6]:
		st.subheader("Do older vs younger ZIP codes have different peak months for events?")
		st.write("ZIPs are split into median-age quartiles and monthly event patterns are averaged within each age bin.")

		age_df = filtered[["zip_code", "median_age"]].dropna().drop_duplicates("zip_code")
		if age_df.empty:
			st.warning("Median age is not available, so age-bin seasonality cannot be computed.")
		else:
			try:
				age_df["age_bin"] = pd.qcut(age_df["median_age"], q=4, labels=["Q1 Youngest", "Q2", "Q3", "Q4 Oldest"], duplicates="drop")
			except ValueError:
				st.warning("Not enough median-age variation to create quartiles.")
				age_df["age_bin"] = "Single Bin"

			zip_month_counts = filtered.groupby(["zip_code", "month"], as_index=False).size().rename(columns={"size": "event_count"})
			merged = zip_month_counts.merge(age_df[["zip_code", "age_bin"]], on="zip_code", how="left")
			merged["age_bin"] = merged["age_bin"].astype(str)
			heat = (
				merged.groupby(["age_bin", "month"], observed=True)["event_count"]
				.mean()
				.reset_index(name="avg_event_count")
			)

			if heat.empty:
				st.warning("No age-bin monthly profile is available for these filters.")
			else:
				heat_pivot = heat.pivot_table(index="age_bin", columns="month", values="avg_event_count", fill_value=0)
				heat_pivot = heat_pivot.reindex(sorted(heat_pivot.columns), axis=1)
				fig_q7 = px.imshow(
					heat_pivot,
					aspect="auto",
					color_continuous_scale=[[0, "#EDF4EE"], [1, PALETTE["primary"]],],
					title="Average Monthly Event Count by Age Quartile",
				)
				st.plotly_chart(fig_q7, use_container_width=True)

				peak_table = heat.loc[heat.groupby("age_bin")["avg_event_count"].idxmax()].copy()
				peak_table["peak_month"] = peak_table["month"].map({
					1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
					7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
				})
				st.dataframe(peak_table[["age_bin", "peak_month", "avg_event_count"]], use_container_width=True)

	# Q8
	with tabs[7]:
		st.subheader("How do predominantly different racial ZIP groups trend in event activity over time?")
		st.write("Select one or more predominant race groups. The chart shows average events per ZIP by year for ZIP codes where the selected race is predominant.")

		if "predominant_race" not in filtered.columns:
			st.warning("Predominant race is not available in the current dataset.")
		else:
			race_lookup = (
				filtered[["zip_code", "predominant_race"]]
				.dropna(subset=["zip_code", "predominant_race"])
				.drop_duplicates(subset=["zip_code"])
			)
			available_races = sorted(race_lookup["predominant_race"].astype(str).unique().tolist())

			selected_races = st.multiselect(
				"Predominant race group(s)",
				options=available_races,
				default=available_races,
			)

			if not selected_races:
				st.warning("Select at least one race group to display the trend line.")
			else:
				selected_zips = race_lookup[race_lookup["predominant_race"].astype(str).isin(selected_races)]["zip_code"]
				q8_df = filtered[filtered["zip_code"].isin(selected_zips)].copy()

				if q8_df.empty:
					st.warning("No events are available for the selected race group(s) with current filters.")
				else:
					zip_year_events = (
						q8_df.groupby(["predominant_race", "year", "zip_code"], as_index=False)
						.agg(events=("event_date", "size"))
					)

					avg_events = (
						zip_year_events.groupby(["predominant_race", "year"], as_index=False)
						.agg(avg_events_per_zip=("events", "mean"))
						.sort_values(["predominant_race", "year"])
					)

					fig_q8 = px.line(
						avg_events,
						x="year",
						y="avg_events_per_zip",
						color="predominant_race",
						markers=True,
						title="Average Events per ZIP by Year (Filtered by Predominant Race)",
						labels={
							"year": "Year",
							"avg_events_per_zip": "Average Events per ZIP",
							"predominant_race": "Predominant Race",
						},
						color_discrete_sequence=[PALETTE["accent"], PALETTE["secondary"], PALETTE["warm"], PALETTE["primary"]],
					)
					_style_chart(fig_q8)
					st.plotly_chart(fig_q8, use_container_width=True)

	# Q9
	with tabs[8]:
		st.subheader("Where is overall event density highest across Chicago?")
		st.write("This map shows event-density hot spots by ZIP code using average geocoded ZIP centroids for Chicago.")

		density_df = (
			filtered[filtered["zip_code"] != "Unknown"]
			.groupby("zip_code", as_index=False)
			.agg(events=("event_date", "size"))
		)

		if density_df.empty:
			st.warning("No ZIP-level events available for the current filters.")
		else:
			max_zip_geocode = st.slider(
				"Max ZIPs to geocode (faster with lower values)",
				min_value=10,
				max_value=80,
				value=35,
				step=5,
			)

			zip_subset = (
				density_df.sort_values("events", ascending=False)
				.head(max_zip_geocode)
			)

			with st.spinner("Building ZIP centroid map for Chicago..."):
				centroids = get_zip_centroids(tuple(sorted(zip_subset["zip_code"].astype(str).unique().tolist())))

			map_df = zip_subset.merge(centroids, on="zip_code", how="left").dropna(subset=["lat", "lon"])

			if map_df.empty:
				st.warning("Could not geocode ZIP centroids for the current selection.")
			else:
				radius = st.slider("Heat radius", min_value=10, max_value=50, value=25, step=1)
				fig_q9 = px.density_mapbox(
					map_df,
					lat="lat",
					lon="lon",
					z="events",
					radius=radius,
					hover_name="zip_code",
					hover_data={"events": True, "lat": ":.4f", "lon": ":.4f"},
					zoom=9,
					center={"lat": 41.8781, "lon": -87.6298},
					mapbox_style="carto-darkmatter",
					color_continuous_scale=[[0.0, "#d8f3dc"], [0.5, PALETTE["accent"]], [1.0, PALETTE["primary"]]],
					title="Chicago Event Density (ZIP-Based)",
				)
				fig_q9.update_layout(
					paper_bgcolor=CHART_BG,
					font={"color": CHART_TEXT},
					margin={"l": 0, "r": 0, "t": 60, "b": 0},
				)
				st.plotly_chart(fig_q9, use_container_width=True)
				if len(density_df) > len(zip_subset):
					st.caption(f"Showing top {len(zip_subset)} ZIPs by event volume for faster rendering.")

	with st.expander("Data & Schema Notes"):
		st.write("Sample of joined analysis dataset (filtered):")
		st.dataframe(filtered.head(50), use_container_width=True)
		st.write("Discovered tables and columns from information_schema:")
		st.dataframe(schema_df, use_container_width=True)


if __name__ == "__main__":
	main()
