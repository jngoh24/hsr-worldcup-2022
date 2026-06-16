import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Foul-Won Model · EPL 2015/16",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    color: #1a1a1a;
}

.stApp {
    background-color: #f7f7f5;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e5e3;
}
section[data-testid="stSidebar"] * {
    color: #1a1a1a !important;
}

/* ── Headers ── */
h1 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-weight: 600;
    font-size: 28px;
    color: #111111;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
h2, h3, h4 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #111111;
    letter-spacing: -0.01em;
}
h3 { font-size: 16px; }
h4 { font-size: 14px; font-weight: 500; color: #444; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e5e5e3;
    border-radius: 4px;
    padding: 16px 20px;
    box-shadow: none;
}
[data-testid="metric-container"] label {
    font-size: 11px;
    font-weight: 500;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'Inter', sans-serif;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px;
    font-weight: 600;
    color: #111 !important;
    font-family: 'Source Serif 4', serif;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 2px solid #e5e5e3;
    gap: 0;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #888;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    color: #111 !important;
    border-bottom: 2px solid #111 !important;
}

/* ── Divider ── */
hr { border-color: #e5e5e3; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e5e3;
    border-radius: 4px;
    background: #fff;
}

/* ── Slider / multiselect labels ── */
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label {
    font-size: 12px;
    font-weight: 500;
    color: #444 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Utility classes ── */
.eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.caption {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: #888;
    margin: 2px 0 10px 0;
}
.kicker {
    font-family: 'Source Serif 4', serif;
    font-size: 13px;
    font-style: italic;
    color: #555;
}
.threshold-badge {
    display: inline-block;
    background: #111;
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    padding: 3px 10px;
    border-radius: 3px;
    letter-spacing: 0.02em;
}
.gained-badge { display:inline-block; background:#e8f5e9; color:#2e7d32;
                font-size:11px; font-weight:600; padding:2px 8px;
                border-radius:3px; font-family:'Inter',sans-serif; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data + model
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_RATE = 0.0183

NUM = ['x','y','dist_to_goal','dist_to_own_goal','dist_to_touchline','central_dist',
       'minute','poss_seq','poss_passes_before','poss_time_elapsed','in_pass_length','pl_enc']
CAT = ['gain_type','pos_role','play_pattern','in_pass_height']
BOOLS = ['under_pressure','on_poss_team','is_home_team','period_2nd']
FEATS = CAT + NUM + BOOLS

@st.cache_data
def load_data():
    players = pd.read_csv(os.path.join(DATA_DIR, "player_stats.csv"))
    teams   = pd.read_csv(os.path.join(DATA_DIR, "team_stats.csv"))
    gains   = pd.read_parquet(os.path.join(DATA_DIR, "gains.parquet"))
    gains['under_pressure'] = gains['under_pressure'].astype(str).str.lower().isin(['true','1']).astype(int)
    enc = players.set_index('player_id')['pl_enc']
    gains['pl_enc'] = gains['player_id'].map(enc).fillna(BASE_RATE)
    return players, teams, gains

@st.cache_resource
def get_model():
    _, _, gains = load_data()
    X = gains[FEATS].copy()
    X['in_pass_height'] = X['in_pass_height'].fillna('None')
    y = gains['won_foul'].values
    pre = ColumnTransformer(
        [('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                                unknown_value=np.nan, encoded_missing_value=np.nan), CAT)],
        remainder='passthrough')
    pipe = Pipeline([('pre', pre),
                     ('hgb', HistGradientBoostingClassifier(
                         max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                         min_samples_leaf=200, l2_regularization=1.0,
                         early_stopping=True, random_state=42))])
    pipe.fit(X, y)
    return pipe

try:
    players_df, teams_df, gains_df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# ─────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────
PLOT_BG    = "#ffffff"
PAPER_BG   = "#f7f7f5"
GRID_COLOR = "#eeeeec"
TEXT_COLOR = "#666666"
ACCENT     = "#1a6b3c"   # Athletic-style dark green
BLUE       = "#1a4b8c"
RED        = "#c0392b"
AMBER      = "#b7791f"
GREEN      = "#1a6b3c"
POS_COLORS = {
    "GK":  "#f0a500",
    "DEF": "#1a4b8c",
    "MID": "#1a6b3c",
    "FWD": "#c0392b",
    "OTH": "#aaaaaa",
    "UNK": "#aaaaaa",
}
POS_ORDER = ["FWD", "MID", "DEF", "GK"]

def base_layout(title="", height=400, xaxis=None, yaxis=None):
    default_axis = dict(gridcolor=GRID_COLOR, showline=False, zeroline=False)
    x = {**default_axis, **(xaxis or {})}
    y = {**default_axis, **(yaxis or {})}
    return dict(
        title=dict(text=title, font=dict(family="Inter", size=13, color="#111111")),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="DM Mono", color=TEXT_COLOR, size=11),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=x,
        yaxis=y,
    )

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="eyebrow" style="margin-bottom:2px;">Foul-Won Model</p>',
                unsafe_allow_html=True)
    st.markdown('<p style="font-family:Inter;font-size:12px;color:#666;margin:0 0 16px 0;">'
                'Premier League 2015/16</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<p class="eyebrow" style="margin-bottom:8px;">League base rate</p>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#f0f0ee;border-left:3px solid {ACCENT};padding:10px 14px;">
        <p style="font-family:Inter;font-size:11px;font-weight:600;color:#888;
                  text-transform:uppercase;letter-spacing:0.06em;margin:0 0 2px 0;">Foul won per possession</p>
        <p style="font-family:'Source Serif 4',serif;font-size:28px;font-weight:600;
                  color:#111;margin:0;">{BASE_RATE*100:.2f}%</p>
        <p style="font-family:Inter;font-size:11px;color:#666;margin:2px 0 0 0;">after a receipt or recovery</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Filters")
    if data_loaded:
        all_teams = sorted(players_df["team"].dropna().unique().tolist())
        selected_teams = st.multiselect("Teams", options=all_teams, default=all_teams,
                                        placeholder="All teams")
        selected_positions = st.multiselect("Position group", options=POS_ORDER,
                                            default=POS_ORDER, placeholder="All positions")
        min_games = st.slider("Min games played", 1, 38, 5)

    st.divider()
    st.markdown('<p style="font-family:Inter;font-size:11px;color:#888;line-height:1.5;">'
                '<strong>Fouls drawn</strong> — a player is fouled on his possession after he '
                'receives or recovers the ball. The Players & Teams tabs count every such foul; '
                'the Pitch & Model tabs use foul-won <em>rates</em> per possession.</p>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="eyebrow" style="margin-bottom:4px;">English Premier League 2015/16 · StatsBomb event data</p>',
            unsafe_allow_html=True)
st.markdown('<h1 style="margin:0 0 4px 0;">Winning Fouls</h1>', unsafe_allow_html=True)
st.markdown(f"<p class='kicker'>Modelling the probability a player wins a foul on his possession "
            f"after receiving or recovering the ball — a league-wide "
            f"<span class='threshold-badge'>{BASE_RATE*100:.2f}%</span> per possession.</p>",
            unsafe_allow_html=True)
st.divider()

if not data_loaded:
    st.error("⚠️ Data files not found. Place `player_stats.csv`, `team_stats.csv`, and "
             "`gains.parquet` in the `data/` folder.")
    st.stop()

# ─────────────────────────────────────────────
# Apply sidebar filters
# ─────────────────────────────────────────────
fp = players_df[
    players_df["team"].isin(selected_teams) &
    players_df["pos_role"].isin(selected_positions) &
    (players_df["games"] >= min_games)
].copy()
ft = teams_df[teams_df["team"].isin(selected_teams)].copy()

# ─────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Players", f"{len(fp):,}")
k2.metric("Total fouls drawn", f"{int(fp['fouls_won'].sum()):,}")
if len(fp):
    top_row = fp.sort_values('fouls_won', ascending=False).iloc[0]
    k3.metric("Most fouled", top_row['player_name'], f"{int(top_row['fouls_won'])} fouls")
    k4.metric("Avg fouls / game", f"{fp['fouls_per_game'].mean():.2f}")
else:
    k3.metric("Most fouled", "—"); k4.metric("Avg fouls / game", "—")
k5.metric("Model ROC-AUC", "0.81", help="Held-out gradient-boosting model; well calibrated")
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Players", "Teams", "Pitch", "Model"])

# ══════════════════════════════════════════════
# TAB 1 — PLAYERS
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Most-fouled players")
    st.markdown(f'<p class="caption">Fouls drawn across the season · colour = position group · '
                f'min {min_games} games</p>', unsafe_allow_html=True)

    metric = st.radio("Rank by", ["Total fouls drawn", "Fouls drawn per game"],
                      horizontal=True, label_visibility="collapsed")
    sort_col = 'fouls_won' if metric.startswith("Total") else 'fouls_per_game'

    col_chart, col_side = st.columns([3, 2])
    with col_chart:
        top_n = st.select_slider("Show top players", options=[10, 15, 20, 25, 30],
                                 value=20, format_func=lambda x: f"Top {x}")
        tp = fp.nlargest(top_n, sort_col)
        labels = tp['player_name'] + "  (" + tp['team'] + ")"
        fig = go.Figure(go.Bar(
            x=tp[sort_col], y=labels, orientation="h",
            marker=dict(color=tp['pos_role'].map(POS_COLORS).fillna("#aaa"),
                        line=dict(width=0), opacity=0.85),
            hovertemplate="<b>%{y}</b><br>" +
                          ("Fouls drawn: %{x:.0f}" if sort_col == 'fouls_won'
                           else "Per game: %{x:.2f}") + "<extra></extra>",
        ))
        fig.update_layout(**base_layout(
            height=max(400, top_n * 22),
            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
            xaxis=dict(title=metric)))
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.markdown("#### Fouls drawn by position")
        grp = (fp.groupby('pos_role')
               .agg(total=('fouls_won', 'sum'), per_game=('fouls_per_game', 'mean'))
               .reindex(POS_ORDER).dropna())
        fig_t = go.Figure(go.Bar(
            x=grp.index, y=grp['total'],
            marker=dict(color=[POS_COLORS[p] for p in grp.index], opacity=0.85),
            text=[f"{int(v):,}" for v in grp['total']], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Total fouls: %{y:,}<extra></extra>"))
        fig_t.update_layout(**base_layout("Total fouls drawn", height=235,
                            yaxis=dict(title=""), xaxis=dict(title="")))
        st.plotly_chart(fig_t, use_container_width=True)

        fig_pg = go.Figure(go.Bar(
            x=grp.index, y=grp['per_game'],
            marker=dict(color=[POS_COLORS[p] for p in grp.index], opacity=0.85),
            text=[f"{v:.2f}" for v in grp['per_game']], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Per game: %{y:.2f}<extra></extra>"))
        fig_pg.update_layout(**base_layout("Avg per game (per player)", height=235,
                             yaxis=dict(title=""), xaxis=dict(title="")))
        st.plotly_chart(fig_pg, use_container_width=True)

    st.markdown("#### Leaderboard")
    tbl = (fp.sort_values(sort_col, ascending=False).head(25)
           [['player_name', 'team', 'position', 'fouls_won', 'games', 'fouls_per_game']]
           .reset_index(drop=True))
    tbl.index = tbl.index + 1
    st.dataframe(tbl, use_container_width=True, height=430,
        column_config={
            'player_name': st.column_config.TextColumn("Player"),
            'team': st.column_config.TextColumn("Team"),
            'position': st.column_config.TextColumn("Position"),
            'fouls_won': st.column_config.NumberColumn("Fouls drawn", format="%d"),
            'games': st.column_config.NumberColumn("Games", format="%d"),
            'fouls_per_game': st.column_config.NumberColumn("Per game", format="%.2f")})

# ══════════════════════════════════════════════
# TAB 2 — TEAMS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### Team foul-drawing")
    st.markdown('<p class="caption">Click a row in the table to drill into a club and see its '
                'top three foul-winners.</p>', unsafe_allow_html=True)

    ft = ft.copy()
    ft['foul_rate'] = (ft['fouls_won_total'] / ft['touches'] * 100).round(2)
    show = (ft.sort_values('fouls_won_total', ascending=False)
            [['team', 'fouls_won_total', 'fouls_per_game', 'games', 'foul_rate']]
            .reset_index(drop=True))

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### League table — fouls drawn")
        event = st.dataframe(
            show, use_container_width=True, height=430, hide_index=True,
            on_select="rerun", selection_mode="single-row",
            column_config={
                'team': st.column_config.TextColumn("Team"),
                'fouls_won_total': st.column_config.NumberColumn("Fouls drawn", format="%d"),
                'fouls_per_game': st.column_config.NumberColumn("Per game", format="%.2f"),
                'games': st.column_config.NumberColumn("Games", format="%d"),
                'foul_rate': st.column_config.NumberColumn("Per 100 touches", format="%.2f")})
        sel = event.selection.rows
        default_team = show.iloc[sel[0]]['team'] if sel else (show.iloc[0]['team'] if len(show) else None)
    with right:
        st.markdown("#### Drill into a club")
        team_list = show['team'].tolist()
        if team_list:
            chosen = st.selectbox("Team", team_list,
                                  index=team_list.index(default_team) if default_team in team_list else 0)
            row = ft[ft['team'] == chosen].iloc[0]
            m1, m2 = st.columns(2)
            m1.metric("Fouls drawn", f"{int(row['fouls_won_total'])}")
            m2.metric("Per game", f"{row['fouls_per_game']:.2f}")
            st.markdown(f'<p class="caption">Top 3 foul-winners · {chosen}</p>', unsafe_allow_html=True)
            top3 = (players_df[players_df['team'] == chosen]
                    .sort_values('fouls_won', ascending=False).head(3)
                    [['player_name', 'position', 'fouls_won', 'fouls_per_game']].reset_index(drop=True))
            top3.index = top3.index + 1
            st.dataframe(top3, use_container_width=True,
                column_config={
                    'player_name': st.column_config.TextColumn("Player"),
                    'position': st.column_config.TextColumn("Position"),
                    'fouls_won': st.column_config.NumberColumn("Fouls drawn", format="%d"),
                    'fouls_per_game': st.column_config.NumberColumn("Per game", format="%.2f")})
        else:
            chosen = None

    st.markdown("#### Fouls drawn per game — all clubs")
    order = ft.sort_values('fouls_per_game', ascending=False)
    colors = [AMBER if t == chosen else ACCENT for t in order['team']]
    fig_team = go.Figure(go.Bar(
        x=order['fouls_per_game'], y=order['team'], orientation="h",
        marker=dict(color=colors, opacity=0.88),
        hovertemplate="<b>%{y}</b><br>Fouls drawn/game: %{x:.2f}<extra></extra>"))
    fig_team.update_layout(**base_layout(
        height=max(400, len(order) * 24),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        xaxis=dict(title="fouls drawn per game")))
    st.plotly_chart(fig_team, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — PITCH
# ══════════════════════════════════════════════
def hex_ramp(t, end=(26, 107, 60)):
    r = int(255 + (end[0] - 255) * t)
    g = int(255 + (end[1] - 255) * t)
    b = int(255 + (end[2] - 255) * t)
    return f"rgba({r},{g},{b},0.88)"

with tab3:
    st.markdown("### Where fouls are won")
    st.markdown('<p class="caption">Foul-won rate per possession across the pitch, by play pattern, '
                'and by how the ball was gained · attack →</p>', unsafe_allow_html=True)

    flt = st.radio("Filter touches", ["All", "Receipts only", "Recoveries only"],
                   horizontal=True, label_visibility="collapsed")
    gp = gains_df
    if flt == "Receipts only":   gp = gains_df[gains_df['gain_type'] == 'receipt']
    elif flt == "Recoveries only": gp = gains_df[gains_df['gain_type'] == 'recovery']

    col_pitch, col_side = st.columns([1.3, 1])
    with col_pitch:
        st.markdown("#### Foul-won rate by pitch location")
        nx, ny = 6, 4
        xed = np.linspace(0, 120, nx + 1); yed = np.linspace(0, 80, ny + 1)
        d = gp.copy()
        d['xb'] = pd.cut(d['x'], xed, labels=False, include_lowest=True)
        d['yb'] = pd.cut(d['y'], yed, labels=False, include_lowest=True)
        cell = d.groupby(['yb', 'xb'])['won_foul'].mean().unstack() * 100
        vmax = np.nanmax(cell.values) if cell.size else 1

        fig_p = go.Figure()
        PITCH_LN = "#c8c8c4"
        fig_p.add_shape(type="rect", x0=0, x1=120, y0=0, y1=80,
                        fillcolor="#f8f8f5", line=dict(color=PITCH_LN, width=1.5), layer="below")
        for j in range(ny):
            for i in range(nx):
                try:
                    rate = cell.loc[j, i]
                except (KeyError, TypeError):
                    rate = np.nan
                if pd.isna(rate): rate = 0.0
                t = rate / vmax if vmax > 0 else 0
                fig_p.add_shape(type="rect", x0=xed[i], x1=xed[i+1], y0=yed[j], y1=yed[j+1],
                                fillcolor=hex_ramp(t), line=dict(color=PITCH_LN, width=0.5), layer="below")
                cx, cy = (xed[i]+xed[i+1])/2, (yed[j]+yed[j+1])/2
                tcol = "#ffffff" if t > 0.55 else "#222222"
                fig_p.add_annotation(x=cx, y=cy, text=f"<b>{rate:.1f}%</b>", showarrow=False,
                                     font=dict(family="Inter", size=11, color=tcol))
        for sh in [
            dict(type="line", x0=60, x1=60, y0=0, y1=80, line=dict(color="#aaaaaa", width=1.2)),
            dict(type="circle", x0=50, x1=70, y0=30, y1=50, line=dict(color="#aaaaaa", width=1)),
            dict(type="rect", x0=0, x1=18, y0=18, y1=62, line=dict(color="#aaaaaa", width=1)),
            dict(type="rect", x0=102, x1=120, y0=18, y1=62, line=dict(color="#aaaaaa", width=1)),
            dict(type="rect", x0=0, x1=120, y0=0, y1=80, line=dict(color="#aaaaaa", width=1.5)),
        ]:
            fig_p.add_shape(**sh)
        for lbl, cx in zip(["DEFENSIVE THIRD", "MIDDLE THIRD", "ATTACKING THIRD"], [20, 60, 100]):
            fig_p.add_annotation(x=cx, y=83.5, text=lbl, showarrow=False,
                                 font=dict(family="Inter", size=9, color="#888"))
        fig_p.add_annotation(x=104, y=-4.5, ax=70, ay=-4.5, xref="x", yref="y", axref="x", ayref="y",
                             showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                             arrowcolor="#888", text="Attack",
                             font=dict(family="Inter", size=9, color="#888"), xanchor="right")
        fig_p.update_layout(
            plot_bgcolor="#f8f8f5", paper_bgcolor=PAPER_BG, height=430,
            margin=dict(l=16, r=16, t=24, b=24),
            xaxis=dict(range=[-4, 124], showgrid=False, zeroline=False, showticklabels=False, showline=False),
            yaxis=dict(range=[-8, 88], showgrid=False, zeroline=False, showticklabels=False,
                       showline=False, scaleanchor="x", scaleratio=1),
            showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)
        st.markdown('<p class="caption">Rate dips inside the box (penalty risk) and peaks through midfield.</p>',
                    unsafe_allow_html=True)

    with col_side:
        st.markdown("#### By play pattern")
        pp = gp.groupby('play_pattern')['won_foul'].agg(['count', 'mean'])
        pp = pp[pp['count'] >= 500].sort_values('mean')
        colors = [AMBER if i == 'From Counter' else ACCENT for i in pp.index]
        fig_pp = go.Figure(go.Bar(
            x=pp['mean']*100, y=pp.index, orientation="h",
            marker=dict(color=colors, opacity=0.88),
            hovertemplate="<b>%{y}</b><br>Foul-won rate: %{x:.1f}%<extra></extra>"))
        fig_pp.add_vline(x=BASE_RATE*100, line=dict(color="#999", width=1, dash="dash"))
        fig_pp.update_layout(**base_layout(height=270,
                             yaxis=dict(tickfont=dict(size=10)),
                             xaxis=dict(title="foul-won rate (%)")))
        st.plotly_chart(fig_pp, use_container_width=True)

        st.markdown("#### Receipt vs recovery")
        gt = gains_df.groupby('gain_type')['won_foul'].mean() * 100
        fig_gt = go.Figure(go.Bar(
            x=["Receipt", "Recovery"],
            y=[gt.get('receipt', 0), gt.get('recovery', 0)],
            marker=dict(color=[ACCENT, AMBER], opacity=0.88),
            text=[f"{gt.get('receipt',0):.2f}%", f"{gt.get('recovery',0):.2f}%"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>"))
        fig_gt.add_hline(y=BASE_RATE*100, line=dict(color="#999", width=1, dash="dash"))
        fig_gt.update_layout(**base_layout(height=210, yaxis=dict(title="rate (%)"), xaxis=dict(title="")))
        st.plotly_chart(fig_gt, use_container_width=True)
        st.markdown('<p class="caption">Recoveries lead to a won foul ~2.3× as often as receipts.</p>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — MODEL
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### Foul-won probability calculator")
    st.markdown('<p class="caption">Set the situation; the gradient-boosting model returns the '
                'calibrated probability of winning a foul on this possession.</p>',
                unsafe_allow_html=True)

    model = get_model()
    pp_opts = sorted(gains_df['play_pattern'].dropna().unique().tolist())

    cfg, out = st.columns([1.25, 1])
    with cfg:
        a, b = st.columns(2)
        gain_type = a.radio("Ball gained by", ["receipt", "recovery"],
                            format_func=lambda s: s.capitalize())
        pos_role = b.selectbox("Player position", POS_ORDER, index=0)
        under_pressure = a.checkbox("Under pressure when gaining the ball", value=True)
        play_pattern = b.selectbox("Play pattern", pp_opts,
            index=pp_opts.index("From Counter") if "From Counter" in pp_opts else 0)

        st.markdown('<p class="eyebrow" style="margin:6px 0 4px;">Player foul-drawing tendency</p>',
                    unsafe_allow_html=True)
        names = ["League average"] + players_df.sort_values('fouls_won', ascending=False)['player_name'].tolist()
        who = st.selectbox("Use a specific player's profile (optional)", names,
                           label_visibility="collapsed")
        pl_enc = BASE_RATE if who == "League average" else \
            float(players_df.loc[players_df['player_name'] == who, 'pl_enc'].iloc[0])
        st.markdown(f'<p class="caption">Tendency used: {pl_enc*100:.2f}% '
                    f'(league avg {BASE_RATE*100:.2f}%)</p>', unsafe_allow_html=True)

        st.markdown('<p class="eyebrow" style="margin:6px 0 4px;">Location on pitch (attack →)</p>',
                    unsafe_allow_html=True)
        lx, ly = st.columns(2)
        x = lx.slider("x — up pitch", 0, 120, 70)
        y = ly.slider("y — width", 0, 80, 40)

        st.markdown('<p class="eyebrow" style="margin:6px 0 4px;">Possession context</p>',
                    unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        passes_before = p1.slider("Passes so far", 0, 15, 0)
        time_elapsed = p2.slider("Secs into poss.", 0, 60, 3)
        minute = p3.slider("Match minute", 0, 95, 60)

        if gain_type == "receipt":
            ip1, ip2 = st.columns(2)
            in_pass_height = ip1.selectbox("Incoming pass height",
                                           ["Ground Pass", "Low Pass", "High Pass"])
            in_pass_length = ip2.slider("Incoming pass length (m)", 0, 60, 18)
        else:
            in_pass_height = "None"; in_pass_length = np.nan

    row = {
        'gain_type': gain_type, 'pos_role': pos_role, 'play_pattern': play_pattern,
        'in_pass_height': in_pass_height, 'x': x, 'y': y,
        'dist_to_goal': float(np.hypot(120 - x, 40 - y)),
        'dist_to_own_goal': float(np.hypot(x, 40 - y)),
        'dist_to_touchline': float(min(y, 80 - y)), 'central_dist': float(abs(y - 40)),
        'minute': minute, 'poss_seq': passes_before, 'poss_passes_before': passes_before,
        'poss_time_elapsed': time_elapsed, 'in_pass_length': in_pass_length, 'pl_enc': pl_enc,
        'under_pressure': int(under_pressure), 'on_poss_team': 1,
        'is_home_team': 1, 'period_2nd': int(minute > 45)}
    prob = float(model.predict_proba(pd.DataFrame([row])[FEATS])[:, 1][0])
    lift = prob / BASE_RATE

    with out:
        st.markdown('<p class="eyebrow" style="margin-bottom:6px;">Predicted probability</p>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#f0f0ee;border-left:3px solid {ACCENT};padding:16px 18px;">
            <p style="font-family:'Source Serif 4',serif;font-size:44px;font-weight:600;
                      color:{ACCENT};margin:0;line-height:1;">{prob*100:.1f}%</p>
            <p style="font-family:Inter;font-size:12px;color:#666;margin:6px 0 0 0;">
               {lift:.1f}× the league average of {BASE_RATE*100:.2f}% per possession</p>
        </div>
        """, unsafe_allow_html=True)

        fig_g = go.Figure(go.Bar(
            x=[prob*100], y=[""], orientation="h",
            marker=dict(color=ACCENT, opacity=0.9),
            hovertemplate="%{x:.2f}%<extra></extra>"))
        fig_g.add_vline(x=BASE_RATE*100, line=dict(color="#999", width=1.2, dash="dash"),
                        annotation_text="league avg", annotation_position="top",
                        annotation_font=dict(size=9, color="#888"))
        fig_g.update_layout(**base_layout(height=120,
                            xaxis=dict(title="probability (%)", range=[0, max(12, prob*100*1.25)]),
                            yaxis=dict(showticklabels=False)))
        fig_g.update_layout(margin=dict(l=10, r=20, t=30, b=40))
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown('<p class="caption">Biggest levers: under-pressure status, the player\'s '
                    'foul-drawing tendency, and counter-attacking play patterns. Toggle "under '
                    'pressure" or switch to a counter to see the swing.</p>', unsafe_allow_html=True)

st.divider()
st.markdown(f'<p class="caption">Data: StatsBomb open event data, EPL 2015/16 · '
            f'323,322 ball receptions/recoveries · league base rate {BASE_RATE*100:.2f}% '
            f'fouls won per possession.</p>', unsafe_allow_html=True)
