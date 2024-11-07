# Player Performance Prediction Model

This Python script uses machine learning to predict basketball player stats, such as points, assists, and rebounds, for an upcoming game based on historical performance data. The model leverages a stacking ensemble of three base regressors—Gradient Boosting, Random Forest, and Linear Regression—with a Linear Regression meta-model to improve prediction accuracy.

## Requirements

- **Python 3.8+**
- **Libraries**: `pandas`, `numpy`, `scikit-learn`

## Features

### `predict_player_points`

Predicts the points a player may score in the next game.

```python
predicted_points, mae, r2 = predict_player_points('player_stats.csv', 'player_averages.csv')
print(predicted_points)
```

### `predict_player_assists`

Predicts the assists a player may have in the next game.

```python
predicted_assists, mae, r2 = predict_player_assists('player_stats.csv', 'player_averages.csv')
print(predicted_assists)
```

### `predict_player_rebounds`

Predicts the rebounds a player may have in the next game.

```python
predicted_rebounds, mae, r2 = predict_player_rebounds('player_stats.csv', 'player_averages.csv')
print(predicted_rebounds)
```
### `calculate_bet_value`

Estimates the expected value (EV) of a bet based on the predicted points, the sportsbook’s line, and the odds. This function helps assess whether a bet on a player's performance has a positive expected value, indicating a potentially profitable opportunity.

#### Parameters

* predicted_points (float): The model's predicted points for the player.
* sportsbook_line (float): The points line provided by the sportsbook.
* mae (float): The mean absolute error from the model, used to approximate the standard deviation.
* sportsbook_odds (float): The odds offered by the sportsbook.

#### Calculation Process

1. Standard Deviation: The function approximates the standard deviation using the MAE.
2. Normal Distribution: A normal distribution is created around the predicted points.
3. Probability Calculations: The probabilities of the player going over or under the sportsbook line are calculated.
4. Expected Value: The expected value (EV) of the bet is computed based on the probabilities and sportsbook odds.

## Data Preprocessing

The model:
1. Merges player box scores and season averages by season.
2. Converts playtime in `MP` to decimal format.
3. Engineers new features (e.g., `FT_FTA`, `FG_FGA`) and recent game weights to prioritize recent performances.
4. Removes rows with missing values in key columns.

## Model Training

The data is split into training and testing sets, with the following steps for each stat:
- The stacking ensemble is trained on engineered features.
- Predictions are made on the test set.
- Model performance metrics (MAE, R²) are output for evaluation.