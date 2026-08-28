"""
app.py
------
Streamlit deployment prototype for BMDS2003 Data Science Assignment.

Trains the Gradient Boosting Classifier (Section 5.4 of the report,
selected as the best model overall in Section 7.1 Selection of Best Model)
directly at app startup from the processed dataset, and lets a user input
a hypothetical player's attributes to predict their EngagementLevel
(Low / Medium / High), along with class probabilities.

Training happens once and is cached (@st.cache_resource), so this stays
fast on every subsequent interaction. Training in-app (rather than loading
a pickle file) avoids scikit-learn version-mismatch errors between the
environment the model was trained in and the environment it is deployed
to (e.g. Streamlit Community Cloud).

Run with:
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Player Engagement Predictor",
    page_icon="🎮",
    layout="centered"
)

MODEL_NAME = "Gradient Boosting Classifier"

difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
engagement_map_inverse = {0: "Low", 1: "Medium", 2: "High"}
genders = ["Male", "Female"]
locations = ["USA", "Europe", "Asia", "Other"]
genres = ["Action", "Strategy", "RPG", "Sports", "Simulation"]

# ----------------------------------------------------------------------
# Train the model (cached — only runs once per app session/restart)
# ----------------------------------------------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("processed_gaming_dataset.csv")

    drop_cols = ["PlayerID", "GameDifficulty", "EngagementLevel", "EngagementLevel_enc", "AgeGroup"]
    X = df.drop(columns=drop_cols)
    y = df["EngagementLevel_enc"]  # 0 = Low, 1 = Medium, 2 = High

    feature_columns = X.columns.tolist()

    model = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=7,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X, y)

    return model, feature_columns

with st.spinner("Loading model (training on first run)..."):
    model, feature_columns = train_model()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🎮 Online Gaming Player Engagement Predictor")
st.markdown(
    f"""
    This tool predicts a player's **Engagement Level** (Low, Medium, or High)
    based on their demographic and in-game behavioural attributes, using the
    **{MODEL_NAME}** trained in Section 5.4 of this project's report —
    selected as the best-performing model in Section 7.1 Selection of Best
    Model (92.74% test accuracy, the highest of the four models compared).
    """
)
st.divider()

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
st.subheader("Enter Player Details")
st.caption(
    "ℹ️ Note: Session activity (right column) drives over 98% of this model's "
    "prediction. Age, Gender, Location, and Game Genre (left column) have minimal "
    "individual impact — see Section 5.4.7 Feature Importance."
)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=15, max_value=49, value=28)
    gender = st.selectbox("Gender", genders)
    location = st.selectbox("Location", locations)
    genre = st.selectbox("Game Genre", genres)
    difficulty = st.selectbox("Game Difficulty", list(difficulty_map.keys()))

with col2:
    play_time_hours = st.slider("Average Play Time per Session (hours)", 0.0, 24.0, 6.0, 0.1)
    sessions_per_week = st.slider("Sessions per Week", 0, 19, 8)
    avg_session_duration = st.slider("Average Session Duration (minutes)", 10, 179, 90)
    player_level = st.slider("Player Level", 1, 99, 45)
    achievements_unlocked = st.slider("Achievements Unlocked", 0, 49, 20)
    in_game_purchases = st.radio("Makes In-Game Purchases?", ["No", "Yes"], horizontal=True)

st.divider()

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if st.button("🔮 Predict Engagement Level", type="primary", use_container_width=True):

    # --- Build the raw input row -------------------------------------
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
    row["GameDifficulty_enc"] = difficulty_map[difficulty]
    row["TotalWeeklyPlayMinutes"] = total_weekly_play_minutes
    row["AchievementRate"] = achievement_rate

    gender_col = f"Gender_{gender}"
    if gender_col in row:
        row[gender_col] = 1

    location_col = f"Location_{location}"
    if location_col in row:
        row[location_col] = 1

    genre_col = f"GameGenre_{genre}"
    if genre_col in row:
        row[genre_col] = 1

    input_df = pd.DataFrame([row])[feature_columns]  # enforce correct column order

    # --- Predict ----------------------------------------------------
    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    pred_label = engagement_map_inverse[pred_class]

    # --- Display result -----------------------------------------------
    st.subheader("Prediction Result")

    color_map = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}
    st.markdown(f"### {color_map[pred_label]} Predicted Engagement Level: **{pred_label}**")

    proba_df = pd.DataFrame({
        "Engagement Level": [engagement_map_inverse[i] for i in range(3)],
        "Probability": pred_proba
    }).sort_values("Engagement Level")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bar_colors = ["#2ECC71" if lvl == pred_label else "#B0BEC5" for lvl in proba_df["Engagement Level"]]
    ax.bar(proba_df["Engagement Level"], proba_df["Probability"], color=bar_colors)
    for i, v in enumerate(proba_df["Probability"]):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Predicted Probability")
    ax.set_title("Class Probability Breakdown")
    st.pyplot(fig)

    with st.expander("See derived features used by the model"):
        st.write(f"**TotalWeeklyPlayMinutes** = {sessions_per_week} × {avg_session_duration} = {total_weekly_play_minutes} minutes/week")
        st.write(f"**AchievementRate** = {achievements_unlocked} ÷ {player_level} = {achievement_rate:.3f}")
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

st.divider()
st.caption(
    "Model: Gradient Boosting Classifier (n_estimators=50, learning_rate=0.05, max_depth=7, "
    "min_samples_split=20, min_samples_leaf=10) — BMDS2003 Data Science Assignment, Section 5.4 & 7.1."
)
