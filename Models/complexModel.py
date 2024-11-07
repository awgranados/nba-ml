import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def predict_player_points(player_stats_file, player_averages_file):
    # Load and preprocess data
    playerStats = pd.read_csv(player_stats_file)
    playerAverages = pd.read_csv(player_averages_file)
    mergedDF = pd.merge(playerStats, playerAverages, on='Season', how='left')

    if 'MP' in mergedDF.columns:
        mergedDF = mergedDF[mergedDF['MP'].apply(lambda x: isinstance(x, str) and ':' in x or isinstance(x, (int, float)))]
        
        def convert_minutes(mp):
            if isinstance(mp, str):
                parts = mp.split(':')
                return int(parts[0]) + int(parts[1]) / 60
            return mp
        
        mergedDF['MP'] = mergedDF['MP'].apply(convert_minutes)

    mergedDF.dropna(subset=['PTS', 'MP', 'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA',
                            'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                            'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 
                            'WS/48', 'OBPM', 'DBPM'], inplace=True)
    mergedDF.reset_index(drop=True, inplace=True)
    
    # Feature engineering
    mergedDF['FT_FTA'] = mergedDF['FT%'] * mergedDF['FTA'] * mergedDF['TS%']
    mergedDF['FG_FGA'] = mergedDF['FG%'] * mergedDF['FGA'] * mergedDF['TS%']
    mergedDF['MP_USG%'] = mergedDF['MP'] * mergedDF['USG%']
    mergedDF['3PA_3P%'] = mergedDF['3PA'] * mergedDF['3P%'] * mergedDF['TS%']
   
    mergedDF['Recency'] = np.linspace(0.1, 1, num=len(mergedDF))
    mergedDF['Weight'] = mergedDF['Recency'] 
    
    features = ['MP', 'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA',
                'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 
                'WS/48', 'OBPM', 'DBPM', 'FT_FTA', 'MP_USG%', 'FG_FGA', '3PA_3P%']
    target = 'PTS'

    X = mergedDF[features]
    y = mergedDF[target]
    weights = mergedDF['Weight']
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=.1, random_state=42
    )

    # Define base models and their hyperparameters
    base_models = [
        ('GradientBoosting', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)),
        ('RandomForest', RandomForestRegressor(n_estimators=500, max_depth=7, random_state=42)),
        ('LinearRegression', LinearRegression())
    ]

    # Meta-model
    meta_model = LinearRegression()

    # Stacking Regressor
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)
    stacking_model.fit(X_train, y_train, sample_weight=weights_train)

    # Make predictions on the test set
    y_pred = stacking_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Stacking Ensemble - MAE: {mae:.2f}, R2: {r2:.2f}")


    # Use rolling averages as the input for the next game prediction
    recent_games = mergedDF[features].tail(10)
    rolling_averages = recent_games.mean()
    pred_stats = pd.DataFrame([rolling_averages])

    predicted_points = stacking_model.predict(pred_stats)
    print(f"Predicted points for the player in their next game: {predicted_points[0]}")


    return predicted_points[0], mae, r2


def predict_player_assists(player_stats_file, player_averages_file):
    # Load and preprocess data
    playerStats = pd.read_csv(player_stats_file)
    playerAverages = pd.read_csv(player_averages_file)
    mergedDF = pd.merge(playerStats, playerAverages, on='Season', how='left')

    if 'MP' in mergedDF.columns:
        mergedDF = mergedDF[mergedDF['MP'].apply(lambda x: isinstance(x, str) and ':' in x or isinstance(x, (int, float)))]
        
        def convert_minutes(mp):
            if isinstance(mp, str):
                parts = mp.split(':')
                return int(parts[0]) + int(parts[1]) / 60
            return mp
        
        mergedDF['MP'] = mergedDF['MP'].apply(convert_minutes)

    mergedDF.dropna(subset=['PTS', 'MP', 'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA',
                            'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                            'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 
                            'WS/48', 'OBPM', 'DBPM'], inplace=True)
    mergedDF.reset_index(drop=True, inplace=True)
    
    # Feature engineering
    mergedDF['MP_USG%'] = mergedDF['MP'] * mergedDF['USG%']
    mergedDF['AST_AST%'] = mergedDF['AST'] * mergedDF['AST%'] 
    mergedDF['AST_TOV%'] = mergedDF['AST'] * mergedDF['TOV%']  
   
    mergedDF['Recency'] = np.linspace(0.1, 1, num=len(mergedDF))
    mergedDF['Weight'] = mergedDF['Recency'] 
    
    features = ['MP', 'PTS' ,'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA',
                'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 
                'WS/48', 'OBPM', 'DBPM', 'MP_USG%', 'AST_AST%', 'AST_TOV%']
    target = 'AST'

    X = mergedDF[features]
    y = mergedDF[target]
    weights = mergedDF['Weight']
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=.1, random_state=42
    )

    # Define base models and their hyperparameters
    base_models = [
        ('GradientBoosting', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)),
        ('RandomForest', RandomForestRegressor(n_estimators=500, max_depth=7, random_state=42)),
        ('LinearRegression', LinearRegression())
    ]

    # Meta-model
    meta_model = LinearRegression()

    # Stacking Regressor
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)
    stacking_model.fit(X_train, y_train, sample_weight=weights_train)

    # Make predictions on the test set
    y_pred = stacking_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Stacking Ensemble - MAE: {mae:.2f}, R2: {r2:.2f}")

    # Use rolling averages as the input for the next game prediction
    recent_games = mergedDF[features].tail(10)
    rolling_averages = recent_games.mean()
    pred_stats = pd.DataFrame([rolling_averages])

    # Make the final prediction
    predicted_assists = stacking_model.predict(pred_stats)
    print(f"Predicted assists for the player in their next game: {predicted_assists[0]}")

    return predicted_assists[0], mae, r2


def predict_player_rebounds(player_stats_file, player_averages_file):
    # Load and preprocess data
    playerStats = pd.read_csv(player_stats_file)
    playerAverages = pd.read_csv(player_averages_file)
    mergedDF = pd.merge(playerStats, playerAverages, on='Season', how='left')

    if 'MP' in mergedDF.columns:
        mergedDF = mergedDF[mergedDF['MP'].apply(lambda x: isinstance(x, str) and ':' in x or isinstance(x, (int, float)))]
        
        def convert_minutes(mp):
            if isinstance(mp, str):
                parts = mp.split(':')
                return int(parts[0]) + int(parts[1]) / 60
            return mp
        
        mergedDF['MP'] = mergedDF['MP'].apply(convert_minutes)

    mergedDF.dropna(subset=['PTS', 'MP', 'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA', 'ORB', 'DRB',
                            'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                            'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 
                            'WS/48', 'OBPM', 'DBPM'], inplace=True)
    mergedDF.reset_index(drop=True, inplace=True)
    
    # Feature engineering
    mergedDF['REB'] = mergedDF['DRB'] + mergedDF['ORB']
    mergedDF['MP_USG%'] = mergedDF['MP'] * mergedDF['USG%']
    mergedDF['DRB_DRB%'] = mergedDF['DRB'] * mergedDF['DRB%'] * mergedDF['TRB%']
    mergedDF['ORB_ORB%'] = mergedDF['ORB'] * mergedDF['ORB%'] * mergedDF['TRB%']   
   
    mergedDF['Recency'] = np.linspace(0.1, 1, num=len(mergedDF))
    mergedDF['Weight'] = mergedDF['Recency'] 
    
    features = ['MP', 'PTS' ,'FG%', '3P%', 'FT%', 'GmSc', '+/-', 'FTA', 'FGA', '3PA',
                'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 
                'WS/48', 'OBPM', 'DBPM', 'MP_USG%', 'DRB_DRB%', 'ORB_ORB%']
    target = 'REB'

    X = mergedDF[features]
    y = mergedDF[target]
    weights = mergedDF['Weight']
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, weights, test_size=.1, random_state=42
    )

    # Define base models and their hyperparameters
    base_models = [
        ('GradientBoosting', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)),
        ('RandomForest', RandomForestRegressor(n_estimators=500, max_depth=7, random_state=42)),
        ('LinearRegression', LinearRegression())
    ]

    # Meta-model
    meta_model = LinearRegression()

    # Stacking Regressor
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)
    stacking_model.fit(X_train, y_train, sample_weight=weights_train)

    # Make predictions on the test set
    y_pred = stacking_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Stacking Ensemble - MAE: {mae:.2f}, R2: {r2:.2f}")

    # Use rolling averages as the input for the next game prediction
    recent_games = mergedDF[features].tail(10)
    rolling_averages = recent_games.mean()
    pred_stats = pd.DataFrame([rolling_averages])
    
    # Make the final prediction
    predicted_assists = stacking_model.predict(pred_stats)
    print(f"Predicted assists for the player in their next game: {predicted_assists[0]}")

    return predicted_assists[0], mae, r2
