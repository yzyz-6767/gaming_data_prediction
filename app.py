"""
app.py
------
BMDS2003 Data Science — Deployment Prototype
Player Engagement Prediction System

A multi-tab Streamlit application covering:
  - Prediction        : interactive engagement-level prediction with model consensus
  - Data Exploration  : dataset overview, distributions, correlation analysis
  - Model Performance : full comparison of all 4 models (Section 6.0 Evaluation)
  - Batch Prediction  : CSV upload for bulk predictions
  - About             : project summary

All 4 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
are trained directly in-app from processed_gaming_dataset.csv using each model's
tuned hyperparameters from Sections 5.1-5.4, and cached so training only happens
once per session.

Run with:
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Player Engagement Prediction",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}
LABEL_COLORS = {"Low": "#E74C3C", "Medium": "#F5A623", "High": "#2ECC71"}
MODEL_COLORS = {
    "Logistic Regression": "#5B84B1",
    "Decision Tree": "#4C9F70",
    "Random Forest": "#D9A441",
    "Gradient Boosting": "#C0504D",
}

DIFFICULTY_MAP = {"Easy": 0, "Medium": 1, "Hard": 2}
GENDERS = ["Male", "Female"]
LOCATIONS = ["USA", "Europe", "Asia", "Other"]
GENRES = ["Action", "Strategy", "RPG", "Sports", "Simulation"]

# ============================================================================
# DATA + MODEL LOADING (cached — runs once per session)
# ============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("processed_gaming_dataset.csv")
    # Reconstruct raw categorical labels from one-hot encoded columns (Section 4.4.1)
    # so they can be used for categorical EDA (crosstabs, bar charts) without needing
    # a second raw CSV file alongside the processed one.
    df["Gender"] = df[["Gender_Female", "Gender_Male"]].idxmax(axis=1).str.replace("Gender_", "")
    df["Location"] = df[["Location_Asia", "Location_Europe", "Location_Other", "Location_USA"]].idxmax(axis=1).str.replace("Location_", "")
    df["GameGenre"] = df[["GameGenre_Action", "GameGenre_RPG", "GameGenre_Simulation", "GameGenre_Sports", "GameGenre_Strategy"]].idxmax(axis=1).str.replace("GameGenre_", "")
    return df

@st.cache_resource
def train_all_models(df):
    drop_cols = ["PlayerID", "GameDifficulty", "EngagementLevel", "EngagementLevel_enc", "AgeGroup", "Gender", "Location", "GameGenre"]
    X = df.drop(columns=drop_cols)
    y = df["EngagementLevel_enc"]
    feature_columns = X.columns.tolist()

    # --- Logistic Regression (Section 5.1: C=0.01, l1_ratio=1.0, solver=saga) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(
        C=0.01, penalty="elasticnet", l1_ratio=1.0, solver="saga",
        max_iter=5000, random_state=42
    )
    lr.fit(X_scaled, y)

    # --- Decision Tree (Section 5.2: gini, max_depth=12, min_samples_leaf=10) ---
    dt = DecisionTreeClassifier(criterion="gini", max_depth=12, min_samples_leaf=10, random_state=42)
    dt.fit(X, y)

    # --- Random Forest (Section 5.3: 150 trees, balanced class weight) ---
    rf = RandomForestClassifier(
        criterion="gini", max_depth=None, max_features="sqrt",
        min_samples_leaf=1, min_samples_split=2, n_estimators=150,
        class_weight="balanced", random_state=42
    )
    rf.fit(X, y)

    # --- Gradient Boosting (Section 5.4: champion model) ---
    gb = GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.05, max_depth=7,
        min_samples_split=20, min_samples_leaf=10, random_state=42
    )
    gb.fit(X, y)

    models = {
        "Logistic Regression": {"model": lr, "scaler": scaler, "needs_scaling": True},
        "Decision Tree": {"model": dt, "scaler": None, "needs_scaling": False},
        "Random Forest": {"model": rf, "scaler": None, "needs_scaling": False},
        "Gradient Boosting": {"model": gb, "scaler": None, "needs_scaling": False},
    }
    return models, feature_columns


@st.cache_resource
def train_models_for_evaluation(df):
    """
    Trains a SEPARATE set of models using ONLY the 70% training split (matching
    the true methodology from Sections 5.1-5.4), so that confusion matrices shown
    in the Model Performance tab reflect genuine, non-leaked test-set performance.
    (The main `models` used for live prediction are trained on the FULL dataset,
    which is correct practice for a deployed model, but would cause data leakage
    if reused here, since the "test set" rows would already have been seen during
    training — Random Forest in particular would show inflated accuracy.)
    """
    drop_cols = ["PlayerID", "GameDifficulty", "EngagementLevel", "EngagementLevel_enc",
                 "AgeGroup", "Gender", "Location", "GameGenre"]
    X = df.drop(columns=drop_cols)
    y = df["EngagementLevel_enc"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    scaler_eval = StandardScaler()
    X_train_scaled = scaler_eval.fit_transform(X_train)
    lr_eval = LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=5000, random_state=42)
    lr_eval.fit(X_train_scaled, y_train)

    dt_eval = DecisionTreeClassifier(criterion="gini", max_depth=12, min_samples_leaf=10, random_state=42)
    dt_eval.fit(X_train, y_train)

    rf_eval = RandomForestClassifier(
        criterion="gini", max_depth=None, max_features="sqrt",
        min_samples_leaf=1, min_samples_split=2, n_estimators=150,
        class_weight="balanced", random_state=42
    )
    rf_eval.fit(X_train, y_train)

    gb_eval = GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.05, max_depth=7,
        min_samples_split=20, min_samples_leaf=10, random_state=42
    )
    gb_eval.fit(X_train, y_train)

    eval_models = {
        "Logistic Regression": {"model": lr_eval, "scaler": scaler_eval, "needs_scaling": True},
        "Decision Tree": {"model": dt_eval, "scaler": None, "needs_scaling": False},
        "Random Forest": {"model": rf_eval, "scaler": None, "needs_scaling": False},
        "Gradient Boosting": {"model": gb_eval, "scaler": None, "needs_scaling": False},
    }
    return eval_models, X_test, y_test


with st.spinner("Training 4 machine learning models on first run — this takes about 60 seconds and only happens once per session..."):
    df = load_data()
    models, feature_columns = train_all_models(df)
    eval_models, X_test_eval, y_test_eval = train_models_for_evaluation(df)
    champion = models["Gradient Boosting"]["model"]

# Pre-compute group means used across multiple tabs
GROUP_MEANS = df.groupby("EngagementLevel")[
    ["SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked", "PlayTimeHours"]
].mean()

# Fixed evaluation results (Section 6.0 Evaluation — verified test-set metrics)
EVAL_RESULTS = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
    "Accuracy (%)": [87.03, 92.21, 92.09, 92.74],
    "Precision (%)": [87.23, 92.21, 92.10, 92.74],
    "Recall (%)": [87.03, 92.21, 92.10, 92.74],
    "F1-Score (%)": [86.99, 92.19, 92.10, 92.72],
    "AUC (%)": [93.60, 94.44, 94.50, 94.49],
})

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def build_feature_row(age, gender, location, genre, difficulty, play_time_hours,
                       sessions_per_week, avg_session_duration, player_level,
                       achievements_unlocked, in_game_purchases):
    total_weekly_play_minutes = sessions_per_week * avg_session_duration
    achievement_rate = achievements_unlocked / player_level if player_level > 0 else 0

    row = {col: 0 for col in feature_columns}
    row["Age"] = age
    row["PlayTimeHours"] = play_time_hours
    row["InGamePurchases"] = 1 if in_game_purchases == "Yes" else 0
    row["SessionsPerWeek"] = sessions_per_week
    row["AvgSessionDurationMinutes"] = avg_session_duration
    row["PlayerLevel"] = player_level
    row["AchievementsUnlocked"] = achievements_unlocked
    row["GameDifficulty_enc"] = DIFFICULTY_MAP[difficulty]
    row["TotalWeeklyPlayMinutes"] = total_weekly_play_minutes
    row["AchievementRate"] = achievement_rate

    gc = f"Gender_{gender}"
    if gc in row:
        row[gc] = 1
    lc = f"Location_{location}"
    if lc in row:
        row[lc] = 1
    gg = f"GameGenre_{genre}"
    if gg in row:
        row[gg] = 1

    return pd.DataFrame([row])[feature_columns]


def predict_all_models(input_df):
    """Returns dict: model_name -> (predicted_label, [p_low, p_med, p_high])"""
    results = {}
    for name, bundle in models.items():
        m = bundle["model"]
        X_in = input_df.copy()
        if bundle["needs_scaling"]:
            X_in = bundle["scaler"].transform(X_in)
        pred_class = m.predict(X_in)[0]
        pred_proba = m.predict_proba(X_in)[0]
        results[name] = (LABEL_MAP[pred_class], pred_proba)
    return results


# ============================================================================
# SIDEBAR — Player Input Form (always visible)
# ============================================================================
with st.sidebar:
    st.title("🎮 Player Engagement")
    st.caption("BMDS2003 Data Science — Group Project")
    st.divider()

    st.subheader("Player Information")
    age = st.slider("Age", 15, 49, 28)
    gender = st.selectbox("Gender", GENDERS)
    location = st.selectbox("Location", LOCATIONS)

    st.divider()
    st.subheader("Game Profile")
    genre = st.selectbox("Game Genre", GENRES)
    difficulty = st.selectbox("Game Difficulty", list(DIFFICULTY_MAP.keys()))
    in_game_purchases = st.radio("Makes In-Game Purchases?", ["No", "Yes"], horizontal=True)

    st.divider()
    st.subheader("Activity Details")
    play_time_hours = st.slider("Avg Play Time / Session (hrs)", 0.0, 24.0, 6.0, 0.1)
    sessions_per_week = st.slider("Sessions per Week", 0, 19, 8)
    avg_session_duration = st.slider("Avg Session Duration (min)", 10, 179, 90)
    player_level = st.slider("Player Level", 1, 99, 45)
    achievements_unlocked = st.slider("Achievements Unlocked", 0, 49, 20)

    st.divider()
    predict_clicked = st.button("🔮 Predict Engagement Risk", type="primary", use_container_width=True)

    st.caption(
        "ℹ️ Session activity (Sessions/Week, Session Duration) drives over 98% of "
        "this model's prediction. Demographics have minimal individual impact — "
        "see Data Exploration tab."
    )

# ============================================================================
# MAIN AREA — TABS
# ============================================================================
tab_predict, tab_explore, tab_perf, tab_batch, tab_about = st.tabs(
    ["🔮 Prediction", "📊 Data Exploration", "🏆 Model Performance", "📁 Batch Prediction", "ℹ️ About"]
)

# ----------------------------------------------------------------------------
# TAB 1: PREDICTION
# ----------------------------------------------------------------------------
with tab_predict:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg,#1f3a4d,#2c5364); padding:24px 28px; border-radius:10px;">
            <h2 style="color:white; margin:0;">Player Engagement Prediction</h2>
            <p style="color:#d0e6f0; margin:6px 0 0 0;">AI-powered engagement prediction using 4 ML models —
            identify at-risk (Low engagement) players before they disengage</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    if not predict_clicked:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("**STEP 1**\n\n**Enter Player Data**\n\nUse the sidebar to input the player's demographic, game, and activity attributes.")
        with c2:
            st.warning("**STEP 2**\n\n**Click Predict**\n\nPress 'Predict Engagement Risk' at the bottom of the sidebar.")
        with c3:
            st.success("**STEP 3**\n\n**View Results**\n\nReview the prediction, model consensus, and player profile comparison.")
        st.write("Awaiting player data...")
    else:
        input_df = build_feature_row(
            age, gender, location, genre, difficulty, play_time_hours,
            sessions_per_week, avg_session_duration, player_level,
            achievements_unlocked, in_game_purchases
        )
        all_preds = predict_all_models(input_df)
        champion_label, champion_proba = all_preds["Gradient Boosting"]
        champion_conf = champion_proba.max()

        # --- Result banner ---
        banner_color = LABEL_COLORS[champion_label]
        banner_text = {
            "High": "HIGH ENGAGEMENT",
            "Medium": "MEDIUM ENGAGEMENT",
            "Low": "LOW ENGAGEMENT — AT RISK"
        }[champion_label]
        n_agree = sum(1 for _, (lbl, _) in all_preds.items() if lbl == champion_label)

        st.markdown(
            f"""
            <div style="background:{banner_color}; padding:20px 28px; border-radius:10px; text-align:center;">
                <h2 style="color:white; margin:0;">{banner_text}</h2>
                <p style="color:white; margin:6px 0 0 0;">
                Gradient Boosting (champion model) predicts <b>{champion_label}</b> engagement with
                {champion_conf*100:.1f}% confidence. {n_agree}/4 models agree.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("Champion model selected in Section 6.0 Evaluation based on highest test-set Accuracy, Precision, Recall, and F1-Score (92.74%).")
        st.write("")

        col_gauge, col_consensus = st.columns(2)

        with col_gauge:
            st.markdown("**Engagement Probability (Champion Model)**")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=champion_conf * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': banner_color},
                    'steps': [
                        {'range': [0, 40], 'color': "#fde2e2"},
                        {'range': [40, 70], 'color': "#fdf0d5"},
                        {'range': [70, 100], 'color': "#d9f2e3"},
                    ],
                },
                title={'text': f"Predicted: {champion_label}"}
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_consensus:
            st.markdown("**Model Consensus**")
            names = list(all_preds.keys())
            confs = [all_preds[n][1].max() * 100 for n in names]
            labels = [all_preds[n][0] for n in names]
            colors = [LABEL_COLORS[l] for l in labels]
            fig_consensus = go.Figure(go.Bar(
                x=confs, y=[f"{n} ({l})" for n, l in zip(names, labels)],
                orientation='h', marker_color=colors,
                text=[f"{c:.1f}%" for c in confs], textposition='outside'
            ))
            fig_consensus.update_layout(
                height=300, margin=dict(t=20, b=10, l=10, r=30),
                xaxis_title="Confidence in Predicted Class (%)", xaxis_range=[0, 105]
            )
            st.plotly_chart(fig_consensus, use_container_width=True)
            st.caption(f"**{n_agree}/4 models predict {champion_label}**")

        # --- Why this prediction (data-grounded explanation, not SHAP) ---
        with st.expander("🔍 Why This Prediction? (Feature Contribution Analysis)"):
            st.markdown(
                "This player's key activity attributes are compared against the dataset's "
                "actual average for each engagement group (Section 3.9), showing which group "
                "this player's behaviour most closely resembles."
            )
            compare_feats = ["SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"]
            player_vals = [sessions_per_week, avg_session_duration, player_level, achievements_unlocked]

            fig_compare = go.Figure()
            for lvl in ["Low", "Medium", "High"]:
                fig_compare.add_trace(go.Bar(
                    name=lvl,
                    x=compare_feats,
                    y=[GROUP_MEANS.loc[lvl, f] for f in compare_feats],
                    marker_color=LABEL_COLORS[lvl],
                    opacity=0.55
                ))
            fig_compare.add_trace(go.Scatter(
                name="This Player", x=compare_feats, y=player_vals,
                mode="markers+lines", marker=dict(size=14, color="black", symbol="diamond"),
                line=dict(color="black", dash="dot")
            ))
            fig_compare.update_layout(barmode="group", height=380, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_compare, use_container_width=True)

            st.markdown(
                f"- **SessionsPerWeek:** this player = {sessions_per_week}/week "
                f"(Low avg = {GROUP_MEANS.loc['Low','SessionsPerWeek']:.1f}, "
                f"Medium avg = {GROUP_MEANS.loc['Medium','SessionsPerWeek']:.1f}, "
                f"High avg = {GROUP_MEANS.loc['High','SessionsPerWeek']:.1f})\n"
                f"- **AvgSessionDurationMinutes:** this player = {avg_session_duration} min "
                f"(Low avg = {GROUP_MEANS.loc['Low','AvgSessionDurationMinutes']:.1f}, "
                f"High avg = {GROUP_MEANS.loc['High','AvgSessionDurationMinutes']:.1f})\n\n"
                "Since `TotalWeeklyPlayMinutes` (Sessions × Duration) alone accounts for ~89% of "
                "the champion model's decision-making (Section 5.4.7), these two attributes are the "
                "primary drivers of this prediction — not demographic attributes."
            )

        # --- Radar chart: player profile vs typical High-engagement player ---
        st.markdown("**Player Profile vs. Typical High-Engagement Player**")
        radar_feats = ["SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked", "PlayTimeHours"]
        radar_max = df[radar_feats].max()
        player_norm = [player_vals[0]/radar_max["SessionsPerWeek"],
                       player_vals[1]/radar_max["AvgSessionDurationMinutes"],
                       player_vals[2]/radar_max["PlayerLevel"],
                       player_vals[3]/radar_max["AchievementsUnlocked"],
                       play_time_hours/radar_max["PlayTimeHours"]]
        high_norm = [GROUP_MEANS.loc["High", f] / radar_max[f] for f in radar_feats]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=high_norm + [high_norm[0]], theta=radar_feats + [radar_feats[0]],
                                             fill='toself', name='Typical High-Engagement Player', line_color="#2ECC71"))
        fig_radar.add_trace(go.Scatterpolar(r=player_norm + [player_norm[0]], theta=radar_feats + [radar_feats[0]],
                                             fill='toself', name='This Player', line_color="#1f3a4d"))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=420,
                                 legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_radar, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2: DATA EXPLORATION
# ----------------------------------------------------------------------------
with tab_explore:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg,#1f3a4d,#2c5364); padding:24px 28px; border-radius:10px;">
            <h2 style="color:white; margin:0;">Data Exploration</h2>
            <p style="color:#d0e6f0; margin:6px 0 0 0;">Exploratory Data Analysis of the Online Gaming Behaviour Dataset</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    st.subheader("Dataset Overview")
    total_players = len(df)
    total_cols_raw = 13
    excluded_cols = 5
    usable_features = len(feature_columns)
    class_counts = df["EngagementLevel"].value_counts(normalize=True) * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Players", f"{total_players:,}")
    m2.metric("Raw Columns", total_cols_raw)
    m3.metric("Excluded Columns", excluded_cols)
    m4.metric("Usable Features", usable_features)
    m5.metric("Medium Engagement %", f"{class_counts['Medium']:.1f}%")

    st.info(
        "**Data Quality Note:** The dataset was already fully curated at the point of collection — "
        "0 missing values, 0 duplicate rows, and 0 statistical outliers (IQR method) were found across "
        "all 40,034 records (Section 3.1-3.2). No imputation or row removal was required."
    )

    with st.expander("Column Details"):
        st.dataframe(df.dtypes.astype(str).rename("Data Type"))

    st.subheader("Engagement Level Distribution")
    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig_pie = px.pie(
            values=df["EngagementLevel"].value_counts().values,
            names=df["EngagementLevel"].value_counts().index,
            hole=0.5,
            color=df["EngagementLevel"].value_counts().index,
            color_discrete_map=LABEL_COLORS
        )
        fig_pie.update_traces(textinfo="label+percent")
        fig_pie.update_layout(height=380, annotations=[dict(text=f'{total_players:,}<br>Players', x=0.5, y=0.5, font_size=16, showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_bar:
        fig_bar = px.bar(
            x=df["EngagementLevel"].value_counts().index,
            y=df["EngagementLevel"].value_counts().values,
            color=df["EngagementLevel"].value_counts().index,
            color_discrete_map=LABEL_COLORS,
            labels={"x": "Engagement Level", "y": "Number of Players"}
        )
        fig_bar.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Feature Correlation (Including Encoded Engagement Level)")
    st.caption("EngagementLevel encoded ordinally (Low=0, Medium=1, High=2) for correlation purposes only (Section 3.13).")
    corr_cols = ["Age", "PlayTimeHours", "SessionsPerWeek", "AvgSessionDurationMinutes",
                 "PlayerLevel", "AchievementsUnlocked", "EngagementLevel_enc"]
    corr_matrix = df[corr_cols].corr().round(3)
    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Feature Distributions by Engagement Level")
    numeric_feature = st.selectbox(
        "Select numeric feature",
        ["Age", "PlayTimeHours", "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"]
    )
    col_hist, col_box = st.columns(2)
    with col_hist:
        fig_hist = px.histogram(df, x=numeric_feature, color="EngagementLevel", barmode="overlay",
                                 color_discrete_map=LABEL_COLORS, opacity=0.6,
                                 title=f"Distribution of {numeric_feature}")
        fig_hist.update_layout(height=380)
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_box:
        fig_box = px.box(df, x="EngagementLevel", y=numeric_feature, color="EngagementLevel",
                          color_discrete_map=LABEL_COLORS, title=f"Box Plot of {numeric_feature}",
                          category_orders={"EngagementLevel": ["Low", "Medium", "High"]})
        fig_box.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Categorical Feature Analysis")
    cat_feature = st.selectbox("Select categorical feature", ["Gender", "Location", "GameGenre", "GameDifficulty"])
    ct = pd.crosstab(df[cat_feature], df["EngagementLevel"], normalize="index") * 100
    ct = ct[["Low", "Medium", "High"]]
    fig_cat = px.bar(ct, barmode="group", color_discrete_map=LABEL_COLORS,
                      labels={"value": "Percentage (%)", cat_feature: cat_feature},
                      title=f"{cat_feature} vs Engagement Level")
    fig_cat.update_layout(height=400)
    st.plotly_chart(fig_cat, use_container_width=True)
    st.caption("Note: differences across categorical groups are small (<2 percentage points), consistent with Section 3.10's finding that demographic attributes have only a weak relationship with engagement.")

    with st.expander("Data Explorer (Full Dataset with Filters)"):
        filter_engagement = st.multiselect("Filter by Engagement Level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
        filtered = df[df["EngagementLevel"].isin(filter_engagement)]
        st.dataframe(filtered.head(500), use_container_width=True)
        st.caption(f"Showing up to 500 of {len(filtered):,} matching rows.")

    with st.expander("Statistical Summary"):
        st.dataframe(df.describe().round(2), use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# ----------------------------------------------------------------------------
with tab_perf:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg,#1f3a4d,#2c5364); padding:24px 28px; border-radius:10px;">
            <h2 style="color:white; margin:0;">Model Performance</h2>
            <p style="color:#d0e6f0; margin:6px 0 0 0;">Evaluation of 4 models (1 baseline + 3 comparison models) — Section 6.0 Evaluation</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    st.subheader("Model Scorecard (Test Set)")
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #c6efce' if v else '' for v in is_max]
    styled = EVAL_RESULTS.set_index("Model").style.apply(highlight_max, axis=0).format("{:.2f}%")
    st.dataframe(styled, use_container_width=True)

    st.success("**Champion Model: Gradient Boosting** — highest Accuracy, Precision, Recall, and F1-Score of all four models (92.74%, 92.74%, 92.74%, 92.72%). Selected for deployment in this application (Section 7.1).")
    st.caption("Baseline Model: Logistic Regression (Multinomial)")

    st.subheader("Performance Comparison (All Metrics)")
    metrics_long = EVAL_RESULTS.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_compare = px.bar(metrics_long, x="Metric", y="Score", color="Model", barmode="group",
                          color_discrete_map=MODEL_COLORS, text_auto=".1f")
    fig_compare.update_layout(height=450, yaxis_range=[80, 100])
    st.plotly_chart(fig_compare, use_container_width=True)

    st.subheader("Confusion Matrices (Test Set, 70/30 Split)")
    st.caption("These matrices use models trained ONLY on the 70% training split (not the full-data models used for live prediction), giving genuine, non-leaked test-set results consistent with Section 6.0 Evaluation.")

    cm_cols = st.columns(4)
    for idx, (name, bundle) in enumerate(eval_models.items()):
        m = bundle["model"]
        X_test_use = X_test_eval.copy()
        if bundle["needs_scaling"]:
            X_test_use = bundle["scaler"].transform(X_test_use)
        y_pred = m.predict(X_test_use)
        cm = confusion_matrix(y_test_eval, y_pred, labels=[0, 1, 2])
        acc = (y_pred == y_test_eval).mean() * 100
        with cm_cols[idx]:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                x=["Low", "Medium", "High"], y=["Low", "Medium", "High"],
                                labels=dict(x="Predicted", y="Actual"))
            fig_cm.update_layout(height=280, title=f"{name} ({acc:.1f}%)", title_font_size=12, coloraxis_showscale=False,
                                  margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Feature Importance Comparison (Tree-Based Models)")
    fi_cols = st.columns(3)
    for idx, name in enumerate(["Decision Tree", "Random Forest", "Gradient Boosting"]):
        m = models[name]["model"]
        importances = pd.Series(m.feature_importances_, index=feature_columns).sort_values(ascending=False).head(8)
        with fi_cols[idx]:
            fig_fi = px.bar(x=importances.values, y=importances.index, orientation="h",
                             color_discrete_sequence=[MODEL_COLORS[name]])
            fig_fi.update_layout(height=320, title=name, title_font_size=13,
                                  margin=dict(t=40, b=10, l=10, r=10), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("All three tree-based models independently agree: TotalWeeklyPlayMinutes, PlayerLevel, and AchievementsUnlocked dominate predictive power (Sections 5.2.7, 5.3.6, 5.4.7).")

# ----------------------------------------------------------------------------
# TAB 4: BATCH PREDICTION
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg,#1f3a4d,#2c5364); padding:24px 28px; border-radius:10px;">
            <h2 style="color:white; margin:0;">Batch Prediction</h2>
            <p style="color:#d0e6f0; margin:6px 0 0 0;">Upload a CSV file to predict engagement levels for multiple players at once</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    st.markdown(
        "Upload a CSV file with the following columns: `Age`, `Gender`, `Location`, `GameGenre`, "
        "`GameDifficulty`, `PlayTimeHours`, `InGamePurchases`, `SessionsPerWeek`, "
        "`AvgSessionDurationMinutes`, `PlayerLevel`, `AchievementsUnlocked`."
    )

    with st.expander("Download a sample template"):
        sample_template = pd.DataFrame({
            "Age": [24, 35], "Gender": ["Male", "Female"], "Location": ["USA", "Europe"],
            "GameGenre": ["Action", "Simulation"], "GameDifficulty": ["Hard", "Easy"],
            "PlayTimeHours": [8.0, 2.0], "InGamePurchases": [1, 0],
            "SessionsPerWeek": [15, 2], "AvgSessionDurationMinutes": [130, 25],
            "PlayerLevel": [70, 10], "AchievementsUnlocked": [35, 3]
        })
        st.dataframe(sample_template, use_container_width=True)
        st.download_button("📥 Download Template CSV", sample_template.to_csv(index=False),
                            "player_template.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_cols = ["Age", "Gender", "Location", "GameGenre", "GameDifficulty",
                              "PlayTimeHours", "InGamePurchases", "SessionsPerWeek",
                              "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"]
            missing = [c for c in required_cols if c not in batch_df.columns]

            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.success(f"Loaded {len(batch_df)} rows. Preview:")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("🔮 Run Batch Prediction", type="primary"):
                    predictions, confidences = [], []
                    for _, row in batch_df.iterrows():
                        input_row = build_feature_row(
                            row["Age"], row["Gender"], row["Location"], row["GameGenre"],
                            row["GameDifficulty"], row["PlayTimeHours"], row["SessionsPerWeek"],
                            row["AvgSessionDurationMinutes"], row["PlayerLevel"],
                            row["AchievementsUnlocked"],
                            "Yes" if row["InGamePurchases"] == 1 else "No"
                        )
                        pred_class = champion.predict(input_row)[0]
                        pred_proba = champion.predict_proba(input_row)[0]
                        predictions.append(LABEL_MAP[pred_class])
                        confidences.append(round(pred_proba.max() * 100, 1))

                    batch_df["Predicted_EngagementLevel"] = predictions
                    batch_df["Confidence_%"] = confidences

                    st.subheader("Batch Prediction Results")
                    st.dataframe(batch_df, use_container_width=True)

                    fig_batch = px.histogram(batch_df, x="Predicted_EngagementLevel", color="Predicted_EngagementLevel",
                                              color_discrete_map=LABEL_COLORS,
                                              category_orders={"Predicted_EngagementLevel": ["Low", "Medium", "High"]})
                    fig_batch.update_layout(height=350, showlegend=False, title="Distribution of Predicted Engagement Levels")
                    st.plotly_chart(fig_batch, use_container_width=True)

                    st.download_button("📥 Download Results CSV", batch_df.to_csv(index=False),
                                        "batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ----------------------------------------------------------------------------
# TAB 5: ABOUT
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg,#1f3a4d,#2c5364); padding:24px 28px; border-radius:10px;">
            <h2 style="color:white; margin:0;">About This Project</h2>
            <p style="color:#d0e6f0; margin:6px 0 0 0;">BMDS2003 Data Science Assignment</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    st.subheader("Project Overview")
    st.markdown(
        """
        This project applies the **CRISP-DM** framework to the **Online Gaming Behaviour Dataset**
        (40,034 player records, Rabie El Kharoua, Kaggle) to predict player **Engagement Level**
        (Low, Medium, or High) based on demographic and behavioural attributes.

        **Business objective:** Enable gaming companies to identify at-risk (Low-engagement) players
        early, so that targeted retention interventions can be applied before players disengage entirely.
        """
    )

    st.subheader("Models Compared")
    st.markdown(
        """
        | Model | Role |
        |---|---|
        | Logistic Regression (Multinomial) | Baseline |
        | Decision Tree Classifier | Comparison |
        | Random Forest Classifier | Comparison |
        | **Gradient Boosting Classifier** | **Champion — selected for deployment** |
        """
    )

    st.subheader("Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", "40,034")
    c2.metric("Raw Attributes", "13")
    c3.metric("Engineered Features", "21")

    st.markdown(
        """
        - **Data Preparation:** 0 missing values, 0 duplicates, 0 outliers found (IQR method).
        - **Feature Engineering:** `TotalWeeklyPlayMinutes` (Sessions × Duration) and `AchievementRate`
          (Achievements ÷ Player Level) were engineered, and became the two most predictive features
          across all tree-based models.
        - **Train/Test Split:** 70/30 stratified split (28,023 training / 12,011 testing records).
        """
    )

    st.subheader("Technology Stack")
    st.markdown(
        """
        - **Data processing:** pandas, NumPy
        - **Machine Learning:** scikit-learn (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
        - **Visualisation:** Plotly, Matplotlib, Seaborn
        - **Deployment:** Streamlit Community Cloud
        """
    )

    st.caption("BMDS2003 Data Science — Tunku Abdul Rahman University of Management and Technology, Academic Year 2026/2027 Semester I")
