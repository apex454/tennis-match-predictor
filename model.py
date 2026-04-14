import pandas as pd

df = pd.read_csv("atp_tennis.csv")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df = df[['player_1', 'player_2', 'rank_1', 'rank_2', 'surface', 'winner']]
df.columns = ['player1', 'player2', 'player1_rank', 'player2_rank', 'surface', 'winner']

df = df.dropna()

df['player1_wins'] = (df['winner'] == df['player1']).astype(int)

df['rank_diff'] = df['player2_rank'] - df['player1_rank']
df['rank_ratio'] = df['player1_rank'] / (df['player2_rank'] + 1)

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

features = ['player1_rank', 'player2_rank', 'rank_diff', 'rank_ratio', 'surface']

X_train = train_df[features].copy()
y_train = train_df['player1_wins']

X_test = test_df[features].copy()
y_test = test_df['player1_wins']

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

X_train['surface'] = encoder.fit_transform(X_train['surface'])
X_test['surface'] = encoder.transform(X_test['surface'])

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=400, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print("Predictions:", predictions[:10])
