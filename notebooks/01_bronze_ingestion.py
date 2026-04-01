# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — Raw GradientSports Ingestion
# MAGIC
# MAGIC Loads GradientSports FIFA World Cup 2022 tracking data from DBFS
# MAGIC and writes raw Delta tables to the DBFS mount.
# MAGIC
# MAGIC **Medallion layer**: Raw → **Bronze** → Silver → Gold
# MAGIC
# MAGIC **Inputs** (DBFS mount)
# MAGIC ```
# MAGIC /mnt/cinqai_outputs/jn_adhoc/hsr/
# MAGIC     tracking_data/    ←  64 × {game_id}.jsonl.bz2
# MAGIC     metadata/         ←  64 × {game_id}.json
# MAGIC     rosters/          ←  64 × {game_id}.json
# MAGIC ```
# MAGIC
# MAGIC **Outputs** (Delta on DBFS)
# MAGIC ```
# MAGIC /mnt/cinqai_outputs/jn_adhoc/hsr/delta/
# MAGIC     tracking_bronze/   ←  all frames, all 64 games, partitioned by game_id
# MAGIC     players_bronze/    ←  player + team metadata
# MAGIC     periods_bronze/    ←  period start/end frame IDs
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## 0 · Install library
# MAGIC
# MAGIC Install `fast-forward-football` via **Cluster → Libraries → PyPI**.
# MAGIC This persists across restarts and is cleaner than %pip in a cell.
# MAGIC If you haven't done that yet, uncomment and run the two lines below once.

# COMMAND ----------

# %pip install fast-forward-football
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 1 · Configuration

# COMMAND ----------

# ---------------------------------------------------------------------------
# Source paths — your DBFS mount
# ---------------------------------------------------------------------------
MOUNT_BASE     = "/dbfs/mnt/cinqai_outputs/jn_adhoc/hsr"
LOCAL_TRACKING = f"{MOUNT_BASE}/tracking_data"
LOCAL_META     = f"{MOUNT_BASE}/metadata"
LOCAL_ROSTER   = f"{MOUNT_BASE}/rosters"

# ---------------------------------------------------------------------------
# Delta output paths — written back to the same mount
# ---------------------------------------------------------------------------
DELTA_BASE             = "dbfs:/mnt/cinqai_outputs/jn_adhoc/hsr/delta"
DELTA_TRACKING_BRONZE  = f"{DELTA_BASE}/tracking_bronze"
DELTA_PLAYERS_BRONZE   = f"{DELTA_BASE}/players_bronze"
DELTA_PERIODS_BRONZE   = f"{DELTA_BASE}/periods_bronze"

# ---------------------------------------------------------------------------
# fastforward settings
# ---------------------------------------------------------------------------
COORDINATES = "cdf"               # metres, origin at pitch centre
ORIENTATION = "static_home_away"  # home team always attacks right (+x)
ONLY_ALIVE  = True                # exclude dead-ball frames
OVERWRITE   = True                # set False to append on re-runs

# ---------------------------------------------------------------------------
# Create Delta output directory
# ---------------------------------------------------------------------------
dbutils.fs.mkdirs(DELTA_BASE)
print(f"Source:  {MOUNT_BASE}")
print(f"Delta:   {DELTA_BASE}")

# COMMAND ----------

# MAGIC %md ## 2 · Discover game files

# COMMAND ----------

import os, re

def list_local(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Path not found: {path}\n"
            f"Check your DBFS mount and folder names."
        )
    return os.listdir(path)

tracking_files = sorted([f for f in list_local(LOCAL_TRACKING) if f.endswith(".jsonl.bz2")])
game_ids       = [re.sub(r"\.jsonl\.bz2$", "", f) for f in tracking_files]

print(f"Found {len(game_ids)} tracking files")
print(f"Sample: {game_ids[:5]}")

# COMMAND ----------

# MAGIC %md ## 3 · Smoke test — one game
# MAGIC
# MAGIC Always test one game before running the full loop.

# COMMAND ----------

from fastforward import gradientsports

def get_roster_path(game_id: str) -> str:
    """Handle both roster filename conventions."""
    plain    = os.path.join(LOCAL_ROSTER, f"{game_id}.json")
    suffixed = os.path.join(LOCAL_ROSTER, f"{game_id}_rosters.json")
    if os.path.exists(plain):
        return plain
    elif os.path.exists(suffixed):
        return suffixed
    raise FileNotFoundError(f"No roster file found for game_id={game_id}")

def load_game(game_id: str):
    """Load one game via fastforward. Returns a TrackingDataset."""
    return gradientsports.load_tracking(
        raw_data        = os.path.join(LOCAL_TRACKING, f"{game_id}.jsonl.bz2"),
        meta_data       = os.path.join(LOCAL_META,     f"{game_id}.json"),
        roster_data     = get_roster_path(game_id),
        layout          = "long",
        coordinates     = COORDINATES,
        orientation     = ORIENTATION,
        only_alive      = ONLY_ALIVE,
        include_game_id = True,
    )

# Test on first game
test_id      = game_ids[0]
test_dataset = load_game(test_id)

print(f"Game {test_id} loaded successfully")
print(f"  Tracking rows : {test_dataset.tracking.height:,}")
print(f"  Players       : {test_dataset.players.height}")
print(f"  Periods       : {test_dataset.periods.height}")
print(f"  Columns       : {test_dataset.tracking.columns}")

# COMMAND ----------

# MAGIC %md ## 4 · Full ingestion loop — all 64 games

# COMMAND ----------

import polars as pl
from datetime import datetime, timezone
import os

PARQUET_STAGING = "/dbfs/mnt/cinqai_outputs/jn_adhoc/hsr/staging"
os.makedirs(PARQUET_STAGING, exist_ok=True)

failed_games  = []
success_count = 0

print(f"Ingesting {len(game_ids)} games — Polars direct to Parquet...\n")

for i, game_id in enumerate(game_ids, 1):
    try:
        dataset     = load_game(game_id)
        ingested_at = datetime.now(timezone.utc).isoformat()

        # Fix timestamp + cast types entirely in Polars — no Spark, no Arrow
        tracking_pl = dataset.tracking.with_columns([
            pl.col("timestamp").dt.total_milliseconds().cast(pl.Int64),
            pl.col("frame_id").cast(pl.Int64),
            pl.col("period_id").cast(pl.Int32),
            pl.col("x").cast(pl.Float32),
            pl.col("y").cast(pl.Float32),
            pl.col("z").cast(pl.Float32),
            pl.lit(ingested_at).alias("_ingested_at"),
        ])

        players_pl = dataset.players.with_columns(
            pl.lit(ingested_at).alias("_ingested_at")
        )
        periods_pl = dataset.periods.with_columns(
            pl.lit(ingested_at).alias("_ingested_at")
        )

        # Write directly to Parquet — pure Polars, no JVM involved
        tracking_pl.write_parquet(f"{PARQUET_STAGING}/{game_id}_tracking.parquet")
        players_pl.write_parquet( f"{PARQUET_STAGING}/{game_id}_players.parquet")
        periods_pl.write_parquet( f"{PARQUET_STAGING}/{game_id}_periods.parquet")

        success_count += 1
        print(f"  [{i:>2}/{len(game_ids)}] ✓  {game_id}  ({tracking_pl.height:,} frames)")

    except Exception as e:
        print(f"  [{i:>2}/{len(game_ids)}] ✗  {game_id}  — {e}")
        failed_games.append({"game_id": game_id, "error": str(e)})

print(f"\nPolars phase done: {success_count} succeeded  |  {len(failed_games)} failed")
if failed_games:
    for g in failed_games:
        print(f"  {g['game_id']}: {g['error']}")

# COMMAND ----------

# MAGIC %md ## 5 · Write Bronze Delta tables

# COMMAND ----------

from pyspark.sql import functions as F

PARQUET_DBFS = "dbfs:/mnt/cinqai_outputs/jn_adhoc/hsr/staging"
OVERWRITE    = True

print("Converting Parquet staging files to Delta...\n")

# Read all tracking parquet files at once — Spark handles this efficiently
tracking_spark = spark.read.parquet(f"{PARQUET_DBFS}/*_tracking.parquet")
players_spark  = spark.read.parquet(f"{PARQUET_DBFS}/*_players.parquet")
periods_spark  = spark.read.parquet(f"{PARQUET_DBFS}/*_periods.parquet")

write_mode = "overwrite" if OVERWRITE else "append"

(tracking_spark.write.format("delta")
    .mode(write_mode).partitionBy("game_id")
    .save(DELTA_TRACKING_BRONZE))

(players_spark.write.format("delta")
    .mode(write_mode).save(DELTA_PLAYERS_BRONZE))

(periods_spark.write.format("delta")
    .mode(write_mode).save(DELTA_PERIODS_BRONZE))

print(f"tracking_bronze: {tracking_spark.count():,} rows")
print(f"players_bronze:  {players_spark.count():,} rows")
print(f"periods_bronze:  {periods_spark.count():,} rows")

# Clean up staging files
import os
staging_local = "/dbfs/mnt/cinqai_outputs/jn_adhoc/hsr/staging"
for f in os.listdir(staging_local):
    os.remove(os.path.join(staging_local, f))
print("\nStaging files cleaned up. Bronze Delta tables ready.")

# COMMAND ----------

# MAGIC %md ## 6 · Validate

# COMMAND ----------

display(
    spark.read.format("delta").load(DELTA_TRACKING_BRONZE)
    .groupBy("game_id")
    .agg(
        F.count("*").alias("n_frames"),
        F.countDistinct("player_id").alias("n_players"),
    )
    .orderBy("game_id")
)

# COMMAND ----------

if failed_games:
    print("Failed games:")
    for g in failed_games:
        print(f"  {g['game_id']}: {g['error']}")
else:
    print("All games ingested successfully. Ready for Notebook 02.")

# COMMAND ----------

game_id = "10502"

df = (
    spark.read.format("delta").load(DELTA_TRACKING_BRONZE)
    .filter(F.col("game_id") == game_id)
)

# Basic shape
print(f"Rows:    {df.count():,}")
print(f"Columns: {df.columns}")

# COMMAND ----------

import json
import os
from pyspark.sql import Row
from datetime import datetime

METADATA_LOCAL = "/dbfs/mnt/cinqai_outputs/jn_adhoc/hsr/metadata"

metadata_rows = []

for filename in os.listdir(METADATA_LOCAL):
    if not filename.endswith(".json"):
        continue
    
    with open(os.path.join(METADATA_LOCAL, filename)) as f:
        data = json.load(f)
    
    # Handle both list and dict format
    match = data[0] if isinstance(data, list) else data
    
    metadata_rows.append({
        "game_id":            str(match["id"]),
        "date":               match.get("date"),
        "season":             match.get("season"),
        "competition_id":     match.get("competition", {}).get("id"),
        "competition_name":   match.get("competition", {}).get("name"),
        "stadium_name":       match.get("stadium", {}).get("name"),
        "home_team_id":       match.get("homeTeam", {}).get("id"),
        "home_team_name":     match.get("homeTeam", {}).get("name"),
        "home_team_short":    match.get("homeTeam", {}).get("shortName"),
        "away_team_id":       match.get("awayTeam", {}).get("id"),
        "away_team_name":     match.get("awayTeam", {}).get("name"),
        "away_team_short":    match.get("awayTeam", {}).get("shortName"),
        "pitch_length":       match.get("stadium", {}).get("pitches", [{}])[0].get("length"),
        "pitch_width":        match.get("stadium", {}).get("pitches", [{}])[0].get("width"),
        "fps":                match.get("fps"),
        "week":               match.get("week"),
    })

metadata_df = spark.createDataFrame(metadata_rows)

(
    metadata_df
    .write.format("delta")
    .mode("overwrite")
    .save(f"{DELTA_BASE}/match_metadata")
)

print(f"Match metadata written: {metadata_df.count()} games")
display(metadata_df.orderBy("date"))

# COMMAND ----------

