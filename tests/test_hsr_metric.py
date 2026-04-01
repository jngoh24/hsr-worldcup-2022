"""
Tests for the relative HSR metric pipeline.
Run with: pytest tests/test_hsr_metric.py -v

All tests use synthetic tracking data so no real data files are needed.
The synthetic data is designed to expose specific edge cases in the
metric definition and pipeline logic.
"""

import polars as pl
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hsr_metric import (
    compute_speed,
    compute_vmax,
    tag_relative_hsr_frames,
    extract_hsr_runs,
    summarise_hsr_per_player,
    run_relative_hsr_pipeline,
    FRAME_RATE_HZ,
    MIN_FRAMES_FOR_VMAX,
)
from hsr_comparison import (
    extract_absolute_hsr_runs,
    compare_definitions,
    INDUSTRY_THRESHOLD_KMH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_player_frames(
    player_id: str,
    speeds_kmh: list,
    game_id: str = "game_1",
    period_id: int = 1,
    start_frame: int = 0,
    frame_rate: float = FRAME_RATE_HZ,
) -> pl.DataFrame:
    """
    Build a synthetic tracking DataFrame for one player with given speeds.

    Converts speed (km/h) back to x positions so compute_speed() can
    re-derive the speed correctly.
    """
    dt = 1.0 / frame_rate
    n = len(speeds_kmh)
    frame_ids = list(range(start_frame, start_frame + n))

    speed_ms = [s / 3.6 for s in speeds_kmh]
    x_positions = [0.0]
    for v in speed_ms[:-1]:
        x_positions.append(x_positions[-1] + v * dt)

    return pl.DataFrame({
        "game_id": [game_id] * n,
        "frame_id": frame_ids,
        "period_id": [period_id] * n,
        "timestamp": [i * int(dt * 1000) for i in range(n)],
        "ball_state": ["alive"] * n,
        "ball_owning_team_id": ["home"] * n,
        "team_id": ["home"] * n,
        "player_id": [player_id] * n,
        "x": [float(x) for x in x_positions],
        "y": [0.0] * n,
        "z": [0.0] * n,
    }).with_columns([
        pl.col("frame_id").cast(pl.UInt32),
        pl.col("x").cast(pl.Float32),
        pl.col("y").cast(pl.Float32),
        pl.col("z").cast(pl.Float32),
    ])


# ---------------------------------------------------------------------------
# compute_speed tests
# ---------------------------------------------------------------------------

class TestComputeSpeed:

    def test_speed_derived_correctly(self):
        """Speed should match the displacement-based calculation."""
        frames = pl.DataFrame({
            "game_id": ["g"] * 3,
            "frame_id": [0, 1, 2],
            "period_id": [1, 1, 1],
            "timestamp": [0, 40, 80],
            "ball_state": ["alive"] * 3,
            "ball_owning_team_id": ["home"] * 3,
            "team_id": ["home"] * 3,
            "player_id": ["p1"] * 3,
            "x": [0.0, 0.4, 0.8],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        }).with_columns([
            pl.col("frame_id").cast(pl.UInt32),
            pl.col("x").cast(pl.Float32),
            pl.col("y").cast(pl.Float32),
            pl.col("z").cast(pl.Float32),
        ])

        result = compute_speed(frames)
        speeds = result["speed_kmh"].to_list()

        assert speeds[0] is None  # first frame has no prior
        assert abs(speeds[1] - 36.0) < 0.5
        assert abs(speeds[2] - 36.0) < 0.5

    def test_frame_gap_produces_null(self):
        """Non-consecutive frame IDs should produce null speed, not inflated value."""
        frames = pl.DataFrame({
            "game_id": ["g"] * 3,
            "frame_id": [0, 1, 10],  # gap at frame 10
            "period_id": [1, 1, 1],
            "timestamp": [0, 40, 400],
            "ball_state": ["alive"] * 3,
            "ball_owning_team_id": ["home"] * 3,
            "team_id": ["home"] * 3,
            "player_id": ["p1"] * 3,
            "x": [0.0, 0.4, 4.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        }).with_columns([
            pl.col("frame_id").cast(pl.UInt32),
            pl.col("x").cast(pl.Float32),
            pl.col("y").cast(pl.Float32),
            pl.col("z").cast(pl.Float32),
        ])

        result = compute_speed(frames)
        assert result["speed_kmh"][2] is None

    def test_speed_clipped_at_45kmh(self):
        """Unrealistically high speeds (noise) should be clipped at 45 km/h."""
        frames = pl.DataFrame({
            "game_id": ["g"] * 2,
            "frame_id": [0, 1],
            "period_id": [1, 1],
            "timestamp": [0, 40],
            "ball_state": ["alive"] * 2,
            "ball_owning_team_id": ["home"] * 2,
            "team_id": ["home"] * 2,
            "player_id": ["p1"] * 2,
            "x": [0.0, 100.0],  # 100m in one frame -> ~9000 km/h
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }).with_columns([
            pl.col("frame_id").cast(pl.UInt32),
            pl.col("x").cast(pl.Float32),
            pl.col("y").cast(pl.Float32),
            pl.col("z").cast(pl.Float32),
        ])

        result = compute_speed(frames)
        assert result["speed_kmh"][1] == 45.0


# ---------------------------------------------------------------------------
# compute_vmax tests
# ---------------------------------------------------------------------------

class TestComputeVmax:

    def test_vmax_uses_percentile_not_peak(self):
        """v_max should be the high percentile, robust to single noise spikes."""
        speeds = [30.0] * 500 + [90.0]
        df = make_player_frames("p1", speeds)
        with_speed = compute_speed(df)

        vmax_df = compute_vmax(with_speed, percentile=99.5)
        vmax = vmax_df.filter(pl.col("player_id") == "p1")["vmax_kmh"][0]

        assert vmax < 35.0, f"v_max {vmax} was pulled up by noise spike"

    def test_low_confidence_flag_for_sparse_player(self):
        """Players with few frames should be flagged as low_confidence."""
        df = make_player_frames("sparse_player", [25.0] * 50)
        with_speed = compute_speed(df)
        vmax_df = compute_vmax(with_speed, min_frames=MIN_FRAMES_FOR_VMAX)

        row = vmax_df.filter(pl.col("player_id") == "sparse_player")
        assert row["low_confidence"][0] == True

    def test_sufficient_frames_not_flagged(self):
        """Players with enough data should not be low_confidence."""
        df = make_player_frames("good_player", [25.0] * 300)
        with_speed = compute_speed(df)
        vmax_df = compute_vmax(with_speed, min_frames=MIN_FRAMES_FOR_VMAX)

        row = vmax_df.filter(pl.col("player_id") == "good_player")
        assert row["low_confidence"][0] == False


# ---------------------------------------------------------------------------
# Core definition tests: relative vs absolute
# ---------------------------------------------------------------------------

class TestRelativeVsAbsoluteDefinition:

    def _make_two_player_scenario(self):
        """
        The canonical example:
        - Player A: vmax=34 km/h. Test section at 22 km/h (64.7% of max) — below 75% threshold.
        - Player B: vmax=26 km/h. Test section at 22 km/h (84.6% of max) — above 75% threshold.

        Structure: jog (5 km/h, 50 frames) | vmax section (300 frames) |
                   jog (5 km/h, 50 frames) | test section (22 km/h, 50 frames)

        Under absolute definition (20 km/h): both count in the test section.
        Under relative definition (75% vmax):
            - Player A: 22/34 = 64.7% -> does NOT count
            - Player B: 22/26 = 84.6% -> DOES count
        """
        a_speeds = [5.0] * 50 + [34.0] * 300 + [5.0] * 50 + [22.0] * 50
        player_a = make_player_frames("player_a", a_speeds)

        b_speeds = [5.0] * 50 + [26.0] * 300 + [5.0] * 50 + [22.0] * 50
        player_b = make_player_frames("player_b", b_speeds)

        return pl.concat([player_a, player_b])

    def test_relative_def_differentiates_players(self):
        """
        Player B (lower vmax, same test speed) should earn an HSR run.
        Player A (higher vmax, same test speed) should NOT earn an HSR run
        in the test section because 22km/h is only 64.7% of their 34km/h max.
        """
        df = self._make_two_player_scenario()
        vmax_df, runs_df, summary_df = run_relative_hsr_pipeline(df, threshold_pct=0.75)

        # Test section starts at frame 400 (50 jog + 300 vmax + 50 jog)
        a_test_runs = runs_df.filter(
            (pl.col("player_id") == "player_a")
            & (pl.col("start_frame_id") >= 400)
        ).height
        b_test_runs = runs_df.filter(
            (pl.col("player_id") == "player_b")
            & (pl.col("start_frame_id") >= 400)
        ).height

        assert a_test_runs == 0, (
            f"Player A at 22km/h (64.7% of 34km/h vmax) should NOT qualify. Got {a_test_runs} runs."
        )
        assert b_test_runs >= 1, (
            f"Player B at 22km/h (84.6% of 26km/h vmax) SHOULD qualify. Got {b_test_runs} runs."
        )

    def test_absolute_def_treats_players_equally(self):
        """Under the absolute definition, both players at 22 km/h should count equally."""
        df = self._make_two_player_scenario()
        with_speed = compute_speed(df)
        runs = extract_absolute_hsr_runs(with_speed, threshold_kmh=20.0)

        player_a_runs = runs.filter(
            (pl.col("player_id") == "player_a")
            & (pl.col("start_frame_id") >= 400)
        ).height
        player_b_runs = runs.filter(
            (pl.col("player_id") == "player_b")
            & (pl.col("start_frame_id") >= 400)
        ).height

        assert player_a_runs == player_b_runs, (
            f"Absolute definition should treat both players the same at 22 km/h. "
            f"Got: A={player_a_runs}, B={player_b_runs}"
        )

    def test_player_below_20kmh_counted_in_relative(self):
        """
        A player with vmax=18 km/h running at 14.5 km/h (80.6% of max) should be
        counted under relative definition but NOT under the 20 km/h absolute rule.
        This is the key 'missed effort' the new definition recovers — a slower
        player working flat out goes uncounted by the industry standard.
        """
        # vmax = 18 km/h — a slow player
        # 75% of 18 = 13.5 km/h threshold
        # Test run: 14.5 km/h for 35 frames (1.4 seconds) — above relative, below absolute
        speeds = [18.0] * 300 + [5.0] * 25 + [14.5] * 35 + [5.0] * 10
        df = make_player_frames("slow_player", speeds)
        with_speed = compute_speed(df)

        vmax_df = compute_vmax(with_speed)
        vmax_val = vmax_df.filter(pl.col("player_id") == "slow_player")["vmax_kmh"][0]
        threshold = vmax_val * 0.75

        # Sanity checks
        assert threshold < 20.0, f"Threshold {threshold} should be below 20 km/h"
        assert 14.5 >= threshold, f"Run speed 14.5 should exceed relative threshold {threshold}"
        assert 14.5 < 20.0, "Run speed should be below the absolute 20 km/h threshold"

        tagged = tag_relative_hsr_frames(with_speed, vmax_df, threshold_pct=0.75)
        relative_runs = extract_hsr_runs(tagged)

        # absolute_runs will be empty since no frames reach 20 km/h in the test section
        absolute_runs = extract_absolute_hsr_runs(with_speed, threshold_kmh=20.0)

        # Filter to just the test run section (frames 325+)
        relative_test = relative_runs.filter(
            (pl.col("player_id") == "slow_player")
            & (pl.col("start_frame_id") >= 325)
        )
        absolute_test = absolute_runs.filter(
            (pl.col("player_id") == "slow_player")
            & (pl.col("start_frame_id") >= 325)
        ) if absolute_runs.height > 0 else absolute_runs

        assert relative_test.height >= 1, \
            "Relative definition should detect the high-effort run below 20 km/h"
        assert absolute_test.height == 0, \
            "Absolute definition should miss this run (speed < 20 km/h)"


# ---------------------------------------------------------------------------
# Duration threshold tests
# ---------------------------------------------------------------------------

class TestDurationThreshold:

    def test_short_burst_excluded(self):
        """A burst lasting < 1 second should not count as an HSR run."""
        speeds = [10.0] * 300 + [30.0] * 20 + [10.0] * 10
        df = make_player_frames("p_burst", speeds)
        vmax_df, runs_df, _ = run_relative_hsr_pipeline(df, min_duration_sec=1.0)

        runs_for_player = runs_df.filter(pl.col("player_id") == "p_burst")
        for row in runs_for_player.iter_rows(named=True):
            assert row["duration_sec"] >= 1.0, \
                f"Run with duration {row['duration_sec']}s shorter than 1s threshold"

    def test_sustained_run_included(self):
        """A run lasting >= 1 second should be included."""
        speeds = [30.0] * 300 + [5.0] * 25 + [30.0] * 60 + [5.0] * 10
        df = make_player_frames("p1", speeds)
        vmax_df, runs_df, _ = run_relative_hsr_pipeline(df, threshold_pct=0.75)

        sustained = runs_df.filter(
            (pl.col("player_id") == "p1")
            & (pl.col("duration_sec") >= 1.0)
        )
        assert sustained.height >= 1, "Should have found at least one sustained run"


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestPipeline:

    def test_full_pipeline_returns_correct_types(self):
        """End-to-end pipeline should return DataFrames with expected schemas."""
        speeds = [30.0] * 400 + [5.0] * 25 + [28.0] * 50 + [5.0] * 25
        df = make_player_frames("player_1", speeds)
        df2 = make_player_frames("player_2", [25.0] * 400 + [5.0] * 25 + [22.0] * 50)
        full_df = pl.concat([df, df2])

        vmax_df, runs_df, summary_df = run_relative_hsr_pipeline(full_df)

        assert isinstance(vmax_df, pl.DataFrame)
        assert isinstance(runs_df, pl.DataFrame)
        assert isinstance(summary_df, pl.DataFrame)

        assert "vmax_kmh" in vmax_df.columns
        assert "duration_sec" in runs_df.columns
        assert "runs_per_game" in summary_df.columns

    def test_no_data_for_ball(self):
        """Ball rows should never appear in HSR run results."""
        player_df = make_player_frames("p1", [30.0] * 400 + [28.0] * 50)
        ball_df = make_player_frames("ball", [40.0] * 450)
        ball_df = ball_df.with_columns(pl.lit("ball").alias("team_id"))

        full_df = pl.concat([player_df, ball_df])
        _, runs_df, summary_df = run_relative_hsr_pipeline(full_df)

        assert "ball" not in runs_df["player_id"].to_list()
        assert "ball" not in summary_df["player_id"].to_list()
