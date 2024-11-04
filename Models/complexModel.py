import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def predict_player_points(player_stats_file, player_averages_file, team_to_predict, test_size=0.1):
    # Load the CSV files
    playerStats = pd.read_csv(player_stats_file)
    playerAverages = pd.read_csv(player_averages_file)

    # Merge DataFrames
    mergedDF = pd.merge(playerStats, playerAverages, on='Season', how='left')

    # Data cleaning and preprocessing
    
    if 'MP' in mergedDF.columns:
        mergedDF = mergedDF[mergedDF['MP'].apply(lambda x: isinstance(x, str) and ':' in x or isinstance(x, (int, float)))]

        def convert_minutes(mp):
            if isinstance(mp, str):
                parts = mp.split(':')
                return int(parts[0]) + int(parts[1]) / 60
            return mp

        mergedDF['MP'] = mergedDF['MP'].apply(convert_minutes)

    # Drop missing values for specific columns
    mergedDF.dropna(subset=['PTS', 'MP', 'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA',
                             'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                             'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 
                             'WS/48', 'OBPM', 'DBPM'], inplace=True)
    mergedDF.reset_index(drop=True, inplace=True)

    mergedDF['FT_FTA'] = mergedDF['FT%'] * mergedDF['FTA'] * mergedDF['TS%']
    mergedDF['FG_FGA'] = mergedDF['FG%'] * mergedDF['FGA']* mergedDF['TS%']
    mergedDF['MP_USG%'] = mergedDF['MP'] * mergedDF['USG%']
    mergedDF['3PA_3P%'] = mergedDF['3PA'] * mergedDF['3P%']* mergedDF['TS%']

    # Create recency and weight columns
    mergedDF['Recency'] = np.logspace(0, -1, num=len(mergedDF), base=10)
    mergedDF['Opponent_Weight'] = mergedDF['Opp'].apply(lambda x: 1.25 if x == team_to_predict else 1)
    mergedDF['Weight'] = mergedDF['Recency'] * mergedDF['Opponent_Weight']

    # Select features and target variable
    features = ['MP', 'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA',
                'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 
                'WS/48', 'OBPM', 'DBPM','FT_FTA','MP_USG%', 'FG_FGA' , '3PA_3P%']
    target = 'PTS'

    # Split the data
    X = mergedDF[features]
    y = mergedDF[target]
    weights = mergedDF['Weight']

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=test_size, random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }

    # Initialize and run GridSearchCV
    model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train, sample_weight=weights_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    #print("Best Parameters:", grid_search.best_params_)

    # Make predictions with the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the best model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #print(f'Best Model Mean Absolute Error: {mae}')
    #print(f'Best Model R-squared: {r2}')

    # Predict points against a specific team using weighted historical data
    team_games = mergedDF[mergedDF['Opp'] == team_to_predict]

    if team_games.empty:
        team_games = mergedDF

    avg_stats = team_games[features].mean()

    pred_stats = pd.DataFrame({
        'MP': avg_stats['MP'],
        'FG%': avg_stats['FG%'],
        '3P%': avg_stats['3P%'],
        'FT%': avg_stats['FT%'],
        'GmSc': avg_stats['GmSc'],
        '+/-': avg_stats['+/-'],
        'FTA': avg_stats['FTA'],
        'FGA': avg_stats['FGA'],
        '3PA': avg_stats['3PA'],
        'PER': avg_stats['PER'] * .8, 
        'TS%': avg_stats['TS%'] * .8,
        '3PAr': avg_stats['3PAr'] * .8,
        'FTr': avg_stats['FTr'] * .8,
        'ORB%': avg_stats['ORB%'] * .8,
        'DRB%': avg_stats['DRB%'] * .8,
        'TRB%': avg_stats['TRB%'] * .8,
        'AST%': avg_stats['AST%'] * .8,
        'STL%': avg_stats['STL%'] * .8,
        'BLK%': avg_stats['BLK%'] * .8,
        'TOV%': avg_stats['TOV%'] * .8,
        'USG%': avg_stats['USG%'] * .8,
        'OWS': avg_stats['OWS'] * .8,
        'DWS': avg_stats['DWS'] * .8,
        'WS': avg_stats['WS'] * .8,
        'WS/48': avg_stats['WS/48'] * .8,
        'OBPM': avg_stats['OBPM'] * .8,
        'DBPM': avg_stats['DBPM'] * .8,
        'FT_FTA': avg_stats['FT%'] * avg_stats['FTA']* avg_stats['TS%'],
        'MP_USG%': avg_stats['MP'] * avg_stats['USG%'],
        'FG_FGA': avg_stats['FG%'] * avg_stats['FGA']* avg_stats['TS%'],
        '3PA_3P%': avg_stats['3PA'] * avg_stats['3P%'] * avg_stats['TS%'],
    }, index=[0])

    # Make the prediction for the specific team
    predicted_points = best_model.predict(pred_stats)
    #print(f'Predicted points for the player against {team_to_predict}: {predicted_points[0]}')

    return predicted_points[0], mae, r2 
