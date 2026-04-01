# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — HSR Metric Tables
# MAGIC
# MAGIC Produces three Gold Delta tables enriched with player names, team names,
# MAGIC and positions — ready for coaching and performance staff analysis.
# MAGIC
# MAGIC **Medallion layer**: Bronze → Silver → **Gold**
# MAGIC
# MAGIC **Outputs**
# MAGIC ```
# MAGIC delta/hsr_runs/            ← one row per run event, with pitch coordinates
# MAGIC delta/hsr_comparison/      ← relative vs absolute definition per player
# MAGIC delta/hsr_player_summary/  ← squad-level overview with team + position
# MAGIC ```
# MAGIC
# MAGIC **Analysis queries included**
# MAGIC - Which position had the most HSR?
# MAGIC - Which team had the most HSR?
# MAGIC - Which teams outperformed the industry standard definition?

# COMMAND ----------

# MAGIC %md ## 1 · Configuration

# COMMAND ----------

DELTA_BASE            = "dbfs:/mnt/cinqai_outputs/jn_adhoc/hsr/delta"
DELTA_TRACKING_SILVER = f"{DELTA_BASE}/tracking_silver"
DELTA_VMAX            = f"{DELTA_BASE}/vmax_per_player"
DELTA_PLAYERS_BRONZE  = f"{DELTA_BASE}/players_bronze"
DELTA_MATCH_META      = f"{DELTA_BASE}/match_metadata"
DELTA_RUNS            = f"{DELTA_BASE}/hsr_runs"
DELTA_COMPARISON      = f"{DELTA_BASE}/hsr_comparison"
DELTA_SUMMARY         = f"{DELTA_BASE}/hsr_player_summary"

SRC_PATH           = "/dbfs/mnt/cinqai_outputs/jn_adhoc/hsr/src"
MIN_DURATION_SEC   = 1.0
FRAME_RATE_HZ      = 25.0
ABSOLUTE_THRESHOLD = 20.0

print(f"Reading from : {DELTA_BASE}")
print(f"Writing to   : {DELTA_BASE}")

# COMMAND ----------

# MAGIC %md ## 2 · Load Silver + v_max tables

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

silver_df = spark.read.format("delta").load(DELTA_TRACKING_SILVER)
vmax_df   = spark.read.format("delta").load(DELTA_VMAX)

print(f"Silver rows : {silver_df.count():,}")
print(f"Players     : {vmax_df.count():,}  "
      f"({vmax_df.filter(~F.col('low_confidence')).count()} high confidence)")

# COMMAND ----------

# MAGIC %md ## 3 · Build enriched player lookup
# MAGIC
# MAGIC Joins player names + positions from players_bronze with
# MAGIC team names from match_metadata.
# MAGIC Deduplicates to one row per player_id.

# COMMAND ----------

match_meta = spark.read.format("delta").load(DELTA_MATCH_META)

# Build team lookup — one row per team_id with full and short name
team_lookup = (
    match_meta.select(
        F.col("home_team_id").alias("team_id"),
        F.col("home_team_name").alias("team_name"),
        F.col("home_team_short").alias("team_short"),
    )
    .union(
        match_meta.select(
            F.col("away_team_id").alias("team_id"),
            F.col("away_team_name").alias("team_name"),
            F.col("away_team_short").alias("team_short"),
        )
    )
    .dropDuplicates(["team_id"])
)

# Player lookup — one row per player, enriched with team name
player_lookup = (
    spark.read.format("delta").load(DELTA_PLAYERS_BRONZE)
    .select("player_id", "name", "team_id", "jersey_number", "position", "is_starter")
    .dropDuplicates(["player_id"])
    .join(team_lookup, on="team_id", how="left")
    .select(
        "player_id",
        F.col("name").alias("player_name"),
        "team_id",
        "team_name",
        "team_short",
        "jersey_number",
        "position",
        "is_starter",
    )
)

print(f"Team lookup  : {team_lookup.count()} teams")
print(f"Player lookup: {player_lookup.count()} players")
display(player_lookup.orderBy("team_name", "position").limit(20))

# COMMAND ----------

# MAGIC %md ## 4 · Extract HSR runs via Pandas UDF
# MAGIC
# MAGIC Groups consecutive is_hsr_frame=True frames into discrete run events.
# MAGIC Minimum duration: 1 second (25 frames at 25 Hz).
# MAGIC Runs are scoped to (player_id, game_id) — cannot span game boundaries.

# COMMAND ----------

import sys
import pandas as pd
import polars as pl

sys.path.insert(0, SRC_PATH)
from hsr_metric import extract_hsr_runs as _extract_hsr_runs

from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, LongType, FloatType
)

RUNS_SCHEMA = StructType([
    StructField("player_id",      StringType(),  True),
    StructField("game_id",        StringType(),  True),
    StructField("period_id",      IntegerType(), True),
    StructField("run_id",         LongType(),    True),
    StructField("start_frame_id", LongType(),    True),
    StructField("end_frame_id",   LongType(),    True),
    StructField("duration_sec",   FloatType(),   True),
    StructField("peak_speed_kmh", FloatType(),   True),
    StructField("mean_speed_kmh", FloatType(),   True),
    StructField("vmax_kmh",       FloatType(),   True),
    StructField("pct_of_vmax",    FloatType(),   True),
    StructField("distance_m",     FloatType(),   True),
    StructField("start_x",        FloatType(),   True),
    StructField("start_y",        FloatType(),   True),
    StructField("end_x",          FloatType(),   True),
    StructField("end_y",          FloatType(),   True),
])

def extract_runs_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    """Extract HSR runs for one (player_id, game_id) group."""
    pl_df  = pl.from_pandas(pdf)
    result = _extract_hsr_runs(pl_df, min_duration_sec=MIN_DURATION_SEC)
    if result.is_empty():
        return pd.DataFrame(columns=[f.name for f in RUNS_SCHEMA.fields])
    return result.to_pandas()

runs_df = (
    silver_df
    .filter(F.col("is_hsr_frame"))
    .groupBy("player_id", "game_id")
    .applyInPandas(extract_runs_pandas, schema=RUNS_SCHEMA)
)

run_count = runs_df.count()
print(f"Extracted {run_count:,} HSR run events")

# COMMAND ----------

# MAGIC %md ## 5 · Write hsr_runs Gold table
# MAGIC
# MAGIC Enriched with player name, team, and position for easy filtering.

# COMMAND ----------

hsr_runs_enriched = (
    runs_df
    .join(
        player_lookup.select("player_id", "player_name", "team_name",
                             "team_short", "position"),
        on="player_id", how="left"
    )
    .join(
        match_meta.select("game_id", "date", "home_team_name", "away_team_name",
                          "competition_name", "stadium_name", "week"),
        on="game_id", how="left"
    )
)

(
    hsr_runs_enriched
    .write.format("delta")
    .mode("overwrite")
    .partitionBy("game_id")
    .save(DELTA_RUNS)
)

print(f"hsr_runs written: {run_count:,} rows → {DELTA_RUNS}")

# COMMAND ----------

# MAGIC %md ## 6 · Definition comparison table
# MAGIC
# MAGIC The core analytical output: who gains and loses HSR credit when switching
# MAGIC from the flat 20 km/h industry standard to the relative 75%-of-vmax definition.

# COMMAND ----------

# Relative run counts (from runs already extracted)
relative_counts = (
    runs_df
    .groupBy("player_id")
    .agg(
        F.count("*").alias("runs_relative"),
        F.countDistinct("game_id").alias("n_games"),
    )
)

# Absolute run counts — pure Spark window, no UDF
MIN_FRAMES_ABS = int(MIN_DURATION_SEC * FRAME_RATE_HZ)
w = Window.partitionBy("player_id", "game_id").orderBy("frame_id")

absolute_counts = (
    silver_df
    .filter(
        F.col("speed_kmh").isNotNull()
        & (F.col("speed_kmh") >= ABSOLUTE_THRESHOLD)
    )
    .withColumn("prev_frame", F.lag("frame_id").over(w))
    .withColumn(
        "is_new_run",
        F.col("prev_frame").isNull()
        | (F.col("frame_id") - F.col("prev_frame") > 1)
    )
    .withColumn("abs_run_id",
        F.sum(F.col("is_new_run").cast("int")).over(w))
    .groupBy("player_id", "game_id", "abs_run_id")
    .agg(F.count("*").alias("n_frames"))
    .filter(F.col("n_frames") >= MIN_FRAMES_ABS)
    .groupBy("player_id")
    .agg(F.count("*").alias("runs_absolute"))
)

# Join, compute delta, enrich with player + team info
comparison_df = (
    vmax_df
    .select("player_id", "vmax_kmh", "speed_threshold_kmh", "low_confidence")
    .filter(~F.col("low_confidence"))
    .join(relative_counts, on="player_id", how="left")
    .join(absolute_counts, on="player_id", how="left")
    .fillna(0, subset=["runs_relative", "runs_absolute", "n_games"])
    .withColumn("run_delta",
        F.col("runs_relative") - F.col("runs_absolute"))
    .withColumn("pct_change",
        F.when(F.col("runs_absolute") > 0,
            F.round(
                (F.col("runs_relative") - F.col("runs_absolute"))
                / F.col("runs_absolute") * 100, 1
            )
        ).otherwise(None))
    .withColumn("category",
        F.when(F.col("run_delta") > 0, "gained")
        .when(F.col("run_delta") < 0, "lost")
        .otherwise("unchanged"))
    .withColumn("runs_per_game_relative",
        F.round(F.col("runs_relative") / F.greatest(F.col("n_games"), F.lit(1)), 2))
    .withColumn("runs_per_game_absolute",
        F.round(F.col("runs_absolute") / F.greatest(F.col("n_games"), F.lit(1)), 2))
    .join(player_lookup, on="player_id", how="left")
    .select(
        "player_id", "player_name", "team_name", "team_short", "position",
        "jersey_number",
        F.round("vmax_kmh", 2).alias("vmax_kmh"),
        F.round("speed_threshold_kmh", 2).alias("threshold_kmh"),
        "n_games",
        "runs_absolute", "runs_relative", "run_delta", "pct_change", "category",
        "runs_per_game_absolute", "runs_per_game_relative",
    )
    .orderBy(F.abs("run_delta"), ascending=False)
)

(
    comparison_df
    .write.format("delta")
    .mode("overwrite")
    .save(DELTA_COMPARISON)
)

print(f"hsr_comparison written → {DELTA_COMPARISON}")
display(
    comparison_df
    .groupBy("category")
    .agg(F.count("*").alias("n_players"))
)

# COMMAND ----------

# MAGIC %md ## 7 · Player summary table

# COMMAND ----------

player_summary = (
    runs_df
    .groupBy("player_id")
    .agg(
        F.count("*").alias("total_runs"),
        F.countDistinct("game_id").alias("games_appeared"),
        F.sum("distance_m").alias("total_hsr_distance_m"),
        F.avg("duration_sec").alias("mean_duration_sec"),
        F.avg("pct_of_vmax").alias("mean_pct_of_vmax"),
        F.max("peak_speed_kmh").alias("tournament_peak_speed_kmh"),
        F.avg("peak_speed_kmh").alias("mean_peak_speed_kmh"),
    )
    .join(
        vmax_df.select("player_id", "vmax_kmh", "speed_threshold_kmh", "low_confidence"),
        on="player_id", how="left",
    )
    .join(player_lookup, on="player_id", how="left")
    .withColumn("runs_per_game",
        F.round(F.col("total_runs") / F.greatest(F.col("games_appeared"), F.lit(1)), 2))
    .withColumn("hsr_distance_per_game_m",
        F.round(F.col("total_hsr_distance_m") / F.greatest(F.col("games_appeared"), F.lit(1)), 1))
    .withColumn("mean_pct_of_vmax_pct",
        F.round(F.col("mean_pct_of_vmax") * 100, 1))
    .select(
        # Identity
        "player_id", "player_name", "team_name", "team_short",
        "position", "jersey_number", "is_starter",
        # Speed profile
        F.round("vmax_kmh", 2).alias("vmax_kmh"),
        F.round("speed_threshold_kmh", 2).alias("threshold_kmh"),
        "low_confidence",
        # Volume
        "games_appeared", "total_runs", "runs_per_game",
        F.round("total_hsr_distance_m", 1).alias("total_hsr_distance_m"),
        "hsr_distance_per_game_m",
        # Quality
        F.round("mean_duration_sec", 2).alias("mean_duration_sec"),
        "mean_pct_of_vmax_pct",
        F.round("tournament_peak_speed_kmh", 2).alias("tournament_peak_speed_kmh"),
        F.round("mean_peak_speed_kmh", 2).alias("mean_peak_speed_kmh"),
    )
    .filter(~F.col("low_confidence"))
    .orderBy("runs_per_game", ascending=False)
)

(
    player_summary
    .write.format("delta")
    .mode("overwrite")
    .save(DELTA_SUMMARY)
)

print(f"hsr_player_summary written → {DELTA_SUMMARY}")
display(player_summary.limit(20))

# COMMAND ----------

# MAGIC %md ## 8 · Gold table audit

# COMMAND ----------

from datetime import datetime

tables = {
    "hsr_runs":           DELTA_RUNS,
    "hsr_comparison":     DELTA_COMPARISON,
    "hsr_player_summary": DELTA_SUMMARY,
}

print(f"Gold table audit — {datetime.utcnow().isoformat()}Z")
print("-" * 65)
for name, path in tables.items():
    n       = spark.read.format("delta").load(path).count()
    history = spark.sql(f"DESCRIBE HISTORY delta.`{path}` LIMIT 1").collect()[0]
    print(f"  {name:<25}  {n:>8,} rows   v{history['version']}   {history['timestamp']}")

# COMMAND ----------

# MAGIC %md ## 9 · Analysis queries
# MAGIC
# MAGIC The three core questions this project answers.

# COMMAND ----------

# MAGIC %md ### Q1 — Which position had the most HSR runs per game?

# COMMAND ----------

display(
    spark.read.format("delta").load(DELTA_SUMMARY)
    .groupBy("position")
    .agg(
        F.round(F.avg("runs_per_game"), 2).alias("avg_runs_per_game"),
        F.round(F.avg("vmax_kmh"), 2).alias("avg_vmax_kmh"),
        F.round(F.avg("threshold_kmh"), 2).alias("avg_threshold_kmh"),
        F.round(F.avg("hsr_distance_per_game_m"), 1).alias("avg_hsr_distance_per_game_m"),
        F.round(F.avg("mean_pct_of_vmax_pct"), 1).alias("avg_intensity_pct_of_vmax"),
        F.count("*").alias("n_players"),
    )
    .orderBy("avg_runs_per_game", ascending=False)
)

# COMMAND ----------

# MAGIC %md ### Q2 — Which team had the most HSR runs per game?

# COMMAND ----------

display(
    spark.read.format("delta").load(DELTA_SUMMARY)
    .groupBy("team_name", "team_short")
    .agg(
        F.round(F.avg("runs_per_game"), 2).alias("avg_runs_per_game_per_player"),
        F.sum("total_runs").alias("total_team_runs"),
        F.round(F.avg("vmax_kmh"), 2).alias("avg_vmax_kmh"),
        F.round(F.avg("hsr_distance_per_game_m"), 1).alias("avg_hsr_distance_per_game_m"),
        F.round(F.avg("tournament_peak_speed_kmh"), 2).alias("avg_peak_speed_kmh"),
        F.count("*").alias("n_players_tracked"),
    )
    .orderBy("avg_runs_per_game_per_player", ascending=False)
)

# COMMAND ----------

# MAGIC %md ### Q3 — Which teams outperformed the industry standard HSR definition?
# MAGIC
# MAGIC A positive `pct_outperformance` means the team's players collectively
# MAGIC earn MORE runs under the relative definition than the flat 20 km/h threshold.
# MAGIC These are teams whose players work hard relative to their own limits
# MAGIC but may be systematically undercounted by the industry standard.

# COMMAND ----------

display(
    spark.read.format("delta").load(DELTA_COMPARISON)
    .groupBy("team_name", "team_short")
    .agg(
        F.sum("runs_relative").alias("total_runs_relative"),
        F.sum("runs_absolute").alias("total_runs_absolute"),
        F.sum("run_delta").alias("total_run_delta"),
        F.round(
            (F.sum("runs_relative") - F.sum("runs_absolute"))
            / F.sum("runs_absolute") * 100, 1
        ).alias("pct_outperformance"),
        F.count(
            F.when(F.col("category") == "gained", 1)
        ).alias("players_who_gained"),
        F.count(
            F.when(F.col("category") == "lost", 1)
        ).alias("players_who_lost"),
        F.count("*").alias("n_players"),
    )
    .orderBy("pct_outperformance", ascending=False)
)

# COMMAND ----------

# MAGIC %md ### Bonus — Top 20 players by runs per game (relative definition)

# COMMAND ----------

display(
    spark.read.format("delta").load(DELTA_SUMMARY)
    .select(
        "player_name", "team_name", "position", "jersey_number",
        "vmax_kmh", "threshold_kmh",
        "games_appeared", "total_runs", "runs_per_game",
        "hsr_distance_per_game_m", "mean_pct_of_vmax_pct",
        "tournament_peak_speed_kmh",
    )
    .orderBy("runs_per_game", ascending=False)
    .limit(20)
)

# COMMAND ----------

# MAGIC %md ### Bonus — Players most affected by definition change

# COMMAND ----------

display(
    spark.read.format("delta").load(DELTA_COMPARISON)
    .select(
        "player_name", "team_name", "position",
        "vmax_kmh", "threshold_kmh",
        "runs_absolute", "runs_relative",
        "run_delta", "pct_change", "category",
    )
    .orderBy(F.abs("run_delta"), ascending=False)
    .limit(30)
)

# COMMAND ----------

# MAGIC %md ### Bonus — Run distribution by pitch zone

# COMMAND ----------

display(
    spark.read.format("delta").load(DELTA_RUNS)
    .withColumn("pitch_zone",
        F.when(F.col("start_x") < -17.5, "Defensive third")
        .when(F.col("start_x") < 17.5,   "Middle third")
        .otherwise("Attacking third"))
    .groupBy("pitch_zone", "position")
    .agg(
        F.count("*").alias("n_runs"),
        F.round(F.avg("peak_speed_kmh"), 2).alias("avg_peak_speed_kmh"),
        F.round(F.avg("pct_of_vmax") * 100, 1).alias("avg_pct_of_vmax"),
        F.round(F.avg("distance_m"), 1).alias("avg_distance_m"),
        F.round(F.avg("duration_sec"), 2).alias("avg_duration_sec"),
    )
    .orderBy("pitch_zone", "n_runs", ascending=[True, False])
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export

# COMMAND ----------

# Export Gold tables to CSV for Streamlit app
EXPORT_PATH = "/dbfs/mnt/cinqai_outputs/jn_adhoc/hsr/streamlit_data"
import os
os.makedirs(EXPORT_PATH, exist_ok=True)

tables = {
    "hsr_player_summary": DELTA_SUMMARY,
    "hsr_comparison":     DELTA_COMPARISON,
    "hsr_runs":           DELTA_RUNS,
}

for name, path in tables.items():
    df = spark.read.format("delta").load(path)
    # Coalesce to single file for easy download
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(f"dbfs:/mnt/cinqai_outputs/jn_adhoc/hsr/streamlit_data/{name}")
    )
    print(f"Exported {name}: {df.count():,} rows")

# COMMAND ----------

# Export match metadata as CSV for the app
(
    spark.read.format("delta").load(f"{DELTA_BASE}/match_metadata")
    .select("game_id", "date", "home_team_name", "home_team_short",
            "away_team_name", "away_team_short", "stadium_name",
            "week", "competition_name")
    .orderBy("date")
    .coalesce(1)
    .write.mode("overwrite")
    .option("header", "true")
    .csv("dbfs:/mnt/cinqai_outputs/jn_adhoc/hsr/streamlit_data/match_metadata")
)

# Export runs with game_id so we can filter by game in the app
# (hsr_runs.csv already has game_id — no new export needed for that)
print("Done")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

DELTA_BASE            = "dbfs:/mnt/cinqai_outputs/jn_adhoc/hsr/delta"
DELTA_TRACKING_SILVER = f"{DELTA_BASE}/tracking_silver"

MIN_DURATION_SEC   = 1.0
FRAME_RATE_HZ      = 25.0
ABSOLUTE_THRESHOLD = 20.0
MIN_FRAMES_ABS     = int(MIN_DURATION_SEC * FRAME_RATE_HZ)

silver_df = spark.read.format("delta").load(DELTA_TRACKING_SILVER)

w = Window.partitionBy("player_id", "game_id").orderBy("frame_id")

absolute_runs = (
    silver_df
    .filter(
        F.col("speed_kmh").isNotNull()
        & (F.col("speed_kmh") >= ABSOLUTE_THRESHOLD)
    )
    .withColumn("prev_frame", F.lag("frame_id").over(w))
    .withColumn(
        "is_new_run",
        F.col("prev_frame").isNull()
        | (F.col("frame_id") - F.col("prev_frame") > 1)
    )
    .withColumn("abs_run_id",
        F.sum(F.col("is_new_run").cast("int")).over(w))
    .groupBy("player_id", "game_id", "abs_run_id")
    .agg(F.count("*").alias("n_frames"))
    .filter(F.col("n_frames") >= MIN_FRAMES_ABS)
    .groupBy("player_id")
    .agg(
        F.count("*").alias("runs_absolute"),
        F.countDistinct("game_id").alias("n_games_absolute"),
    )
)

# Export to CSV
(
    absolute_runs
    .coalesce(1)
    .write.mode("overwrite")
    .option("header", "true")
    .csv("dbfs:/mnt/cinqai_outputs/jn_adhoc/hsr/streamlit_data/hsr_absolute_runs")
)

print(f"Absolute run counts exported: {absolute_runs.count()} players")
display(absolute_runs.orderBy("runs_absolute").limit(10))