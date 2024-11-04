# Algorithm with weighted features for recency and specific team games
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the CSV file
df = pd.read_csv('PlayerStats/portemi01.csv')

# Display the first few rows and the columns to understand the structure
print(df.head())
print("Columns in DataFrame:", df.columns)

# Remove rows where the player did not play ('Inactive' or 'Did Not Dress') if applicable
if 'MP' in df.columns:
    df = df[df['MP'].apply(lambda x: isinstance(x, str) and ':' in x or isinstance(x, (int, float)))]

    # Convert 'MP' (minutes played) into total minutes as a float
    def convert_minutes(mp):
        if isinstance(mp, str):
            parts = mp.split(':')
            return int(parts[0]) + int(parts[1]) / 60
        return mp  # Return the value directly if it's already numeric

    df['MP'] = df['MP'].apply(convert_minutes)

# Convert percentage columns to decimal format
percentage_columns = ['FG%', '3P%', 'FT%']
for col in percentage_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0  # Convert to decimal format

# Replace any missing percentage values ('FG%', '3P%', 'FT%') with 0
df[['FG%', '3P%', 'FT%']] = df[['FG%', '3P%', 'FT%']].fillna(0)

# Ensure 'GmSc' and '+/-' are numeric (if they're not already)
df['FGA'] = pd.to_numeric(df['FGA'], errors='coerce')
df['FTA'] = pd.to_numeric(df['FTA'], errors='coerce')
df['3PA'] = pd.to_numeric(df['3PA'], errors='coerce')
df['GmSc'] = pd.to_numeric(df['GmSc'], errors='coerce')
df['+/-'] = pd.to_numeric(df['+/-'], errors='coerce')

# Drop rows with any missing values in the target column or features
df.dropna(subset=['PTS', 'MP', 'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA'], inplace=True)

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

# Create a 'recency' column to account for how recent the game is (using row number as a proxy for date)
df['Recency'] = np.exp(-np.linspace(0, 10, num=len(df)))  # Exponential decay

# Assign extra weight to games against the selected team
team_to_predict = "LAC"  # Example team abbreviation
df['Opponent_Weight'] = df['Opp'].apply(lambda x: 3 if x == team_to_predict else 1)

# Create a combined weight column (recency + opponent weight)
df['Weight'] = df['Recency'] * df['Opponent_Weight']

# Select features and target variable
features = ['MP', 'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA']
target = 'PTS' 

# Split the data into training and testing sets (keeping the weight column for use later)
X = df[features]
y = df[target]
weights = df['Weight']  # Keep track of weights

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the weights
model.fit(X_train, y_train, sample_weight=weights_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Now, to predict points against a specific team using weighted historical data
team_games = df[df['Opp'] == team_to_predict]  # Use 'Opp' column for opponent

# If no games found for the opponent, use all available data
if team_games.empty:
    team_games = df

# Select only numeric columns for averaging
numeric_cols = ['MP', 'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA']
avg_stats = team_games[numeric_cols].mean()

# Prepare a DataFrame for prediction
pred_stats = pd.DataFrame({
    'MP': avg_stats['MP'],
    'FG%': avg_stats['FG%'],
    '3P%': avg_stats['3P%'],
    'FT%': avg_stats['FT%'],
    'GmSc': avg_stats['GmSc'],
    '+/-': avg_stats['+/-'],
    'FTA': avg_stats['FTA'],
    'FGA': avg_stats['FGA'],
    '3PA': avg_stats['3PA']
}, index=[0])  # Need to reshape for prediction

# Make the prediction
predicted_points = model.predict(pred_stats)

print(f'Predicted points for the player against {team_to_predict}: {predicted_points[0]}')
