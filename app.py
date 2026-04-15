import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Tennis Predictor", layout="centered")

st.title("🎾 Tennis Match Predictor")
st.write("Predict match outcome using ML")

df = pd.read_csv("atp_tennis.csv")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df = df[['player_1','player_2','rank_1','rank_2','surface','winner']]
df.columns = ['player1','player2','player1_rank','player2_rank','surface','winner']

df = df.dropna()

df['player1_wins'] = (df['winner'] == df['player1']).astype(int)

df['rank_diff'] = df['player2_rank'] - df['player1_rank']
df['rank_ratio'] = df['player1_rank'] / (df['player2_rank'] + 1)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

features = ['player1_rank','player2_rank','rank_diff','rank_ratio','surface']

X = df[features].copy()
y = df['player1_wins']

encoder = LabelEncoder()
X['surface'] = encoder.fit_transform(X['surface'])

model = RandomForestClassifier(n_estimators=400, random_state=42)
model.fit(X, y)

st.subheader("Enter Match Details")

col1, col2 = st.columns(2)

with col1:
    player1 = st.text_input("Player 1 Name", "Player A")
    rank1 = st.number_input("Player 1 Rank", 1, 1000, 10)

with col2:
    player2 = st.text_input("Player 2 Name", "Player B")
    rank2 = st.number_input("Player 2 Rank", 1, 1000, 20)

surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])

if st.button("Predict Winner"):
    surface_encoded = encoder.transform([surface])[0]

    rank_diff = rank2 - rank1
    rank_ratio = rank1 / (rank2 + 1)

    input_data = [[rank1, rank2, rank_diff, rank_ratio, surface_encoded]]

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.divider()

    if prediction == 1:
        st.success(f"🏆 {player1} is likely to WIN")
        st.write(f"Confidence: {round(prob[1]*100, 2)}%")
    else:
        st.error(f"🏆 {player2} is likely to WIN")
        st.write(f"Confidence: {round(prob[0]*100, 2)}%")
