# Relative High-Speed Running — FIFA World Cup 2022

**A data science portfolio project built to asses player effort.**

Redefining high-speed running (HSR) as a relative metric — >= 80% of a player's personal v-max sustained for >= 1 second — rather than the industry-standard flat 20 km/h threshold. Built on GradientSports broadcast tracking data from the 2022 FIFA Men's World Cup, processed through an Azure Databricks medallion pipeline, and deployed as an interactive Streamlit dashboard.

**Live dashboard:** https://hsr-worldcup-2022-gfveg8myjojscddwtja6kc.streamlit.app/

---

## The problem with the industry standard

This project was directly inspired by the [GradientSports MLS 2024 HSR analysis](https://www.blog.fc.pff.com/blog/pff-fc-mlsphysical-2024) — published by the same provider whose open tracking data powers this project. Their analysis defined a high-speed run as any effort above a flat 20 km/h threshold. That definition is the industry standard. This project uses the same data source to challenge it.

The flat 20 km/h threshold creates a systematic bias: it undercounts high-effort runs from players whose physical ceiling sits below 25 km/h, and overcredits effort from players who coast above 20 km/h without working hard relative to their own limits.

A midfielder running at 22 km/h when their v-max is 23 km/h is working at 96% of their capacity. A centre-back at 22 km/h when their v-max is 30 km/h is barely exerting themselves. The flat threshold treats these identically.

The relative definition fixes this by anchoring each player's threshold to their own physiology — making HSR a genuine measure of individual effort rather than a speed cut-off that favours physically faster players.

**Key finding:** At elite international level, the majority of outfield players have a relative threshold that sits close to or below the 20 km/h industry standard. The flat threshold systematically undercounts high-effort runs — it never overcounts effort, it only misses it.

---

## Data

**Source:** [GradientSports FIFA World Cup 2022 open tracking data](https://gradientai.io/), loaded via the [`fast-forward-football`](https://github.com/UnravelSports/fast-forward-football) Python package.

- 64 games · 32 teams · 82 million tracking frames
- 25 Hz broadcast tracking — x/y position per player, 25 times per second
- Pitch coordinate system: origin at centre, x in [-52.5, 52.5] metres, y in [-34, 34] metres

---

## Dashboard

**Live:** https://hsr-worldcup-2022-gfveg8myjojscddwtja6kc.streamlit.app/

Built with Streamlit and Plotly across seven tabs. Every chart recalculates dynamically as filters change — no page reloads, no pre-aggregated snapshots.

**Sidebar filters apply globally across all tabs simultaneously:**

- **Threshold slider (60-95%)** — the centrepiece of the dashboard. Drag it and every chart, ranking, heatmap, and comparison table recalculates from the raw run data in real time using each player's personal v-max. At 80% the definition reflects this project's chosen threshold; raising it to 90%+ captures only near-maximum efforts.
- **Team filter** — multiselect across all 32 teams; narrows all analyses to the selected subset.
- **Position filter** — toggle between GK, DEF, MID, and FWD to isolate position-specific patterns.
- **Minimum games** — excludes players with limited appearances to avoid small-sample noise.
- **Match filter** — scopes the entire dashboard to a single game, turning it into a per-match analysis tool.

| Tab | Content |
|-----|---------|
| **Players** | Top N bar chart coloured by position · speed vs volume scatter · per-player 30-zone pitch heatmap · full player table |
| **Teams** | HSR volume ranking · speed/volume scatter · HSR distance chart · team summary table |
| **Positions** | Bar chart · radar chart · position table |
| **Definition Comparison** | Delta histogram · threshold scatter vs 20 km/h line · team outperformance bar · full player comparison table |
| **Pitch Map** | 9-zone heatmap of run start positions |
| **Match Analysis** | Per-match top performers · team comparison · position breakdown · pitch heatmap |
| **Tournament Phases** | Group Stage vs Knockout KPI bars · week-by-week trend lines · position breakdown by round |

**The per-player zone heatmap** (Players tab) is the most analytically distinctive feature — select any player and see a 30-zone grid (6 columns x 5 rows) showing where their HSR runs start across the pitch, with run count and percentage per zone. Directly replicates and extends the per-player analysis from the GradientSports MLS article.

---

## Pipeline

```
GradientSports World Cup 2022 open tracking data
    -> fast-forward-football (Python / Polars)
        -> Azure Databricks -- Bronze / Silver / Gold Delta Lake
            -> Gold metric tables (CSV)
                -> Streamlit Community Cloud dashboard
```

### Bronze — Raw ingestion

Loads all 64 games and writes tracking frames, player rosters, period metadata, and match metadata to Delta Lake.

**Design decisions worth noting:**

The main challenge was memory. Converting 1.3M+ rows per game from Polars to Spark in a single operation on an 8.8 GB single-node cluster caused OOM errors. The fix was to separate the two compute layers entirely: Polars writes each game to Parquet on DBFS independently (no JVM involved), then Spark reads all 64 Parquet files in a single batch and writes Delta. This eliminated the memory issue completely.

```
hsr/delta/
  tracking_bronze/     82,231,917 rows  partitioned by game_id
  players_bronze/           3,230 rows
  periods_bronze/             134 rows
  match_metadata/              64 rows
```

### Silver — Feature engineering

Derives per-frame speed from x/y displacement and tags HSR frames for each player.

**Speed computation:** Euclidean distance between frame N and N-1 divided by the frame interval (40ms at 25 Hz), converted to km/h. Distributed via `applyInPandas` on (player_id, game_id) partitions — Polars handles the per-player math, Spark handles the distribution.

**v-max estimation:** p99.9 of all observed speeds pooled across all 64 games. Tournament-level pooling gives a more stable estimate than per-game, particularly important for broadcast tracking where players are regularly off-camera.

p99.9 was chosen after investigation revealed that p99.5 — the initial choice — produced a median v-max of only 20 km/h across the dataset. This is an artefact of broadcast tracking: players are only captured when visible to the camera, meaning the vast majority of tracked frames show them walking or jogging. At 25 Hz across 64 games, p99.5 corresponds to a speed exceeded in only 1 in 200 frames — too conservative for sparse tracking data. p99.9 (1 in 1,000 frames) produces a median v-max of 24.7 km/h, which better reflects genuine peak capability while remaining robust to noise spikes.

Players with fewer than 250 tracked frames are flagged `low_confidence` and excluded from all downstream analysis.

**HSR threshold:** `speed_kmh >= vmax_kmh * threshold` where threshold defaults to 0.80. The dashboard makes this configurable from 60% to 95% in real time.

```
hsr/delta/
  tracking_silver/     82M rows + speed_kmh + is_hsr_frame  partitioned by game_id
  vmax_per_player/     one row per player
```

### Gold — Metric tables

Groups consecutive HSR frames into discrete run events (minimum 1 second / 25 frames). Enriches all tables with player names, team names, and positions joined from the Bronze roster and metadata tables. Also computes the side-by-side comparison against the industry standard flat threshold.

```
hsr/delta/
  hsr_runs/            one row per run event
                         -- player, game, period, start/end frame
                         -- start_x, start_y, end_x, end_y
                         -- duration_sec, peak_speed_kmh, pct_of_vmax, distance_m
  hsr_comparison/      relative vs absolute run counts per player
  hsr_player_summary/  tournament-level aggregation per player
```

---

## Metric definition

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Relative threshold | 80% of personal v-max | Captures sustained high-intensity effort below sprint level |
| v-max estimation | p99.9 across all 64 games | Robust to noise spikes; appropriate for sparse broadcast tracking data |
| Minimum duration | 1 second (25 frames at 25 Hz) | Filters positional noise; consistent with industry convention |
| Min frames for v-max | 250 (~10 seconds tracked) | Excludes players with insufficient data for a reliable estimate |
| Speed clip ceiling | 45 km/h | Guards against frame-gap teleportation artefacts in broadcast tracking |

**Position rollup:**

| Detail codes | Group |
|-------------|-------|
| GK | GK |
| CB, LCB, RCB, LB, RB, LWB, RWB | DEF |
| CDM, CM, LCM, RCM, CAM, LAM, RAM, LM, RM | MID |
| LW, RW, LF, RF, CF, SS, ST | FWD |

---

## Core metric code

`src/hsr_metric.py` is pure Polars — no Spark dependency, fully testable locally.

```python
compute_speed(df)              # adds speed_kmh from x/y displacement
compute_vmax(df, percentile)   # p99.9 per player across all 64 games
tag_relative_hsr_frames(df)    # flags frames above relative threshold
extract_hsr_runs(df)           # groups consecutive frames into run events
summarise_hsr_per_player(df)   # tournament-level aggregation
```

**Tests:** 13 unit tests in `tests/test_hsr_metric.py` covering canonical two-player scenarios, missed runs below threshold, duration gating, and ball exclusion. All passing.

---

## Infrastructure

| Component | Detail |
|-----------|--------|
| Cluster | Azure Databricks · Standard_D4ds_v5 · single-node · 8.8 GB RAM |
| Runtime | Databricks 17.2.x · Spark 3.5 · Python 3.11 |
| Storage | Azure DBFS · Delta Lake |
| Compute | PySpark + Polars · `applyInPandas` for distributed UDFs |
| Dashboard | Streamlit Community Cloud |

---

## Repo contents

```
hsr-worldcup-2022/
├── notebooks/
│   ├── 01_bronze_ingestion.py
│   ├── 02_silver_features.py
│   ├── 03_gold_metrics.py
│   └── 04_gk_investigation.py
├── src/
│   ├── hsr_metric.py
│   └── hsr_comparison.py
├── tests/
│   └── test_hsr_metric.py
├── app/
│   ├── app.py
│   └── data/
│       ├── match_metadata.csv
│       ├── hsr_player_summary.csv
│       ├── hsr_comparison.csv
│       ├── hsr_runs.csv
│       └── hsr_absolute_runs.csv
├── .streamlit/
│   └── config.toml
└── requirements.txt
```

---

*Data: GradientSports FIFA World Cup 2022 open dataset.*  
*Built for the USSF Data Scientist application.*
