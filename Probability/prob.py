from scipy.stats import norm
import numpy as np


def calculate_bet_value(predicted_points, sportsbook_line, mae, sportsbook_odds):
    # Calculate standard deviation
    std_dev = mae * (np.pi / 2) ** 0.5  # Approximation from MAE

    # Create normal distribution
    distribution = norm(loc=predicted_points, scale=std_dev)

    # Probability of going over/under the sportsbook line
    prob_over = 1 - distribution.cdf(sportsbook_line)
    prob_under = distribution.cdf(sportsbook_line)

    # Calculate implied probability from odds
    implied_prob = 1 / (sportsbook_odds + 1)

    # Determine expected value
    expected_value = (prob_over * sportsbook_odds) - ((1 - prob_over) * 1)
    
    return prob_over, implied_prob, expected_value

