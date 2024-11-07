import pandas as pd

def checkAboveLine(player_stats_file, stat, line):
    player_stats = pd.read_csv(player_stats_file)
    
    # Count occurrences where the stat is above the line
    playerAboveLine = (player_stats[stat] > line).sum()

    percentageAboveLine = playerAboveLine/player_stats.shape[0]
    
    return playerAboveLine, percentageAboveLine, player_stats.shape[0]
