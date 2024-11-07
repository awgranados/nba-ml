from Models.complexModel import predict_player_points, predict_player_assists, predict_player_rebounds
from Probability.prob import calculate_bet_value
from scipy.stats import norm
import numpy as np

player_stats_file = 'PlayerStats/butleji01.csv'
player_averages_file = 'PlayerAverages/butleji01.csv'
'''
#example usage of predict_player_points
predicted_points, mae, r2 = predict_player_points(player_stats_file, player_averages_file)
print(f'Best Model Mean Absolute Error: {mae}')
print(f'Best Model R-squared: {r2}')
print(f'Predicted points for the player: {predicted_points}')

sportsbook_line = 18.5
sportsbook_odds = 1.1 

prob_over, implied_prob, expected_value = calculate_bet_value(predicted_points, sportsbook_line, mae, sportsbook_odds)

print(f"Probability of scoring over {sportsbook_line} points: {prob_over:.2%}")
print(f"Implied Probability by sportsbook: {implied_prob:.2%}")
print(f"Expected Value of Bet: {expected_value}")
'''
'''
#example usage of predict_player_assists
predicted_assists, mae, r2 = predict_player_assists(player_stats_file, player_averages_file)
print(f'Best Model Mean Absolute Error: {mae}')
print(f'Best Model R-squared: {r2}')
print(f'Predicted assists for the player: {predicted_assists}')

sportsbook_line = 5.5
sportsbook_odds = 1.1 

prob_over, implied_prob, expected_value = calculate_bet_value(predicted_assists, sportsbook_line, mae, sportsbook_odds)

print(f"Probability of recording over {sportsbook_line} assists: {prob_over:.2%} ")
print(f"Implied Probability by sportsbook: {implied_prob:.2%}")
print(f"Expected Value of Bet: {expected_value}")
'''

#example usage of predict_player_rebounds
predicted_rebounds, mae, r2 = predict_player_rebounds(player_stats_file, player_averages_file)
print(f'Best Model Mean Absolute Error: {mae}')
print(f'Best Model R-squared: {r2}')
print(f'Predicted rebounds for the player: {predicted_rebounds}')

sportsbook_line = 5.5
sportsbook_odds = 1.1 

prob_over, implied_prob, expected_value = calculate_bet_value(predicted_rebounds, sportsbook_line, mae, sportsbook_odds)

print(f"Probability of recording over {sportsbook_line} rebounds: {prob_over:.2%}")
print(f"Implied Probability by sportsbook: {implied_prob:.2%}")
print(f"Expected Value of Bet: {expected_value}")