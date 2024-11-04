from Models.complexModel import predict_player_points
from Probability.prob import calculate_bet_value
from scipy.stats import norm
import numpy as np

player_stats_file = 'PlayerStats/westbru01.csv'
player_averages_file = 'PlayerAverages/westbru01.csv'
team_to_predict = 'UTA'

predicted_points, mae, r2 = predict_player_points(player_stats_file, player_averages_file, team_to_predict)
print(f'Best Model Mean Absolute Error: {mae}')
print(f'Best Model R-squared: {r2}')
print(f'Predicted points for the player against {team_to_predict}: {predicted_points}')

sportsbook_line = 15.5
sportsbook_odds = 1.1 

prob_over, implied_prob, expected_value = calculate_bet_value(predicted_points, sportsbook_line, mae, sportsbook_odds)

print(f"Probability of scoring over {sportsbook_line}: {prob_over:.2%} points")
print(f"Implied Probability by sportsbook: {implied_prob:.2%}")
print(f"Expected Value of Bet: {expected_value}")
