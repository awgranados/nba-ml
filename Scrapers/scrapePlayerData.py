from basketball_reference_web_scraper import client
import csv
from datetime import datetime

# Define the CSV file and headers
playerid = 'butleji01'
start_year = 2023
csv_file = '../PlayerStats/butleji01.csv'
csv_columns = [
    'Rk', 'G', 'Date', 'Tm', 'Opp', 'MP', 
    'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 
    'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF' ,
    'PTS', 'GmSc', '+/-', 'Season'
]

# Team abbreviations mapping
team_abbreviations = {
    'ATLANTA HAWKS': 'ATL',
    'BOSTON CELTICS': 'BOS',
    'BROOKLYN NETS': 'BKN',
    'CHARLOTTE HORNETS': 'CHA',
    'CHICAGO BULLS': 'CHI',
    'CLEVELAND CAVALIERS': 'CLE',
    'DALLAS MAVERICKS': 'DAL',
    'DENVER NUGGETS': 'DEN',
    'DETROIT PISTONS': 'DET',
    'GOLDEN STATE WARRIORS': 'GSW',
    'HOUSTON ROCKETS': 'HOU',
    'INDIANA PACERS': 'IND',
    'LOS ANGELES CLIPPERS': 'LAC',
    'LOS ANGELES LAKERS': 'LAL',
    'MEMPHIS GRIZZLIES': 'MEM',
    'MIAMI HEAT': 'MIA',
    'MILWAUKEE BUCKS': 'MIL',
    'MINNESOTA TIMBERWOLVES': 'MIN',
    'NEW ORLEANS PELICANS': 'NOP',
    'NEW YORK KNICKS': 'NYK',
    'OKLAHOMA CITY THUNDER': 'OKC',
    'ORLANDO MAGIC': 'ORL',
    'PHILADELPHIA 76ERS': 'PHI',
    'PHOENIX SUNS': 'PHX',
    'PORTLAND TRAIL BLAZERS': 'POR',
    'SACRAMENTO KINGS': 'SAC',
    'SAN ANTONIO SPURS': 'SAS',
    'TORONTO RAPTORS': 'TOR',
    'UTAH JAZZ': 'UTA',
    'WASHINGTON WIZARDS': 'WAS'
}

# Define the seasons and their corresponding date ranges
Seasons = {
    '2009-10': ('2009-10-27', '2010-04-14'),
    '2010-11': ('2010-10-26', '2011-04-13'),
    '2011-12': ('2011-12-25', '2012-04-26'),
    '2012-13': ('2012-10-30', '2013-04-17'),
    '2013-14': ('2013-10-29', '2014-04-16'),
    '2014-15': ('2014-10-28', '2015-04-15'),
    '2015-16': ('2015-10-27', '2016-04-13'),
    '2016-17': ('2016-10-25', '2017-04-12'),
    '2017-18': ('2017-10-17', '2018-04-11'),
    '2018-19': ('2018-10-16', '2019-04-10'),
    '2019-20': ('2019-10-22', '2020-03-11'),
    '2020-21': ('2020-12-22', '2021-05-16'),
    '2021-22': ('2021-10-19', '2022-04-10'),
    '2022-23': ('2022-10-18', '2023-04-09'),
    '2023-24': ('2023-10-24', '2024-04-14'),
    '2024-25': ('2024-10-22', '2025-04-13'),
    '2025-26': ('2025-10-21', '2026-04-12')
}


# Save stats to CSV
try:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()

        # Loop through all seasons that LeBron played
        for season_year in range(start_year, 2026):  # Adjust the range as needed
            # Scrape regular season stats for each season
            player_stats = client.regular_season_player_box_scores(
                player_identifier=playerid,
                season_end_year=season_year
            )

            for game_number, stat in enumerate(player_stats, start=1):
                # Calculate game date
                game_date = stat['date']
                game_date_str = game_date.strftime("%Y-%m-%d")  # Format the date for CSV

                # Determine season based on game date
                season = None
                for season_key, (start_date, end_date) in Seasons.items():
                    if start_date <= game_date_str <= end_date:
                        season = season_key
                        break

                # Determine opponent abbreviation
                opponent = stat['opponent'].value
                opponent_abbreviation = team_abbreviations.get(opponent, opponent)
                team_abbreviation = team_abbreviations.get(stat['team'].value, stat['team'].value)  # Team abbreviation

                # Prepare data for CSV
                writer.writerow({
                    'Rk': game_number,
                    'G': game_number,  # Game number
                    'Date': game_date_str,  # Game date
                    'Tm': team_abbreviation,  # Team abbreviation
                    'Opp': opponent_abbreviation,  # Opponent abbreviation
                    'MP': f"{stat['seconds_played'] // 60}:{stat['seconds_played'] % 60:02}",  # Format minutes played
                    'FG': stat['made_field_goals'],
                    'FGA': stat['attempted_field_goals'],
                    'FG%': f"{(stat['made_field_goals'] / stat['attempted_field_goals'] * 100 if stat['attempted_field_goals'] > 0 else 0):.3f}",
                    '3P': stat['made_three_point_field_goals'],
                    '3PA': stat['attempted_three_point_field_goals'],
                    '3P%': f"{(stat['made_three_point_field_goals'] / stat['attempted_three_point_field_goals'] * 100 if stat['attempted_three_point_field_goals'] > 0 else 0):.3f}",
                    'FT': stat['made_free_throws'],
                    'FTA': stat['attempted_free_throws'],
                    'FT%': f"{(stat['made_free_throws'] / stat['attempted_free_throws'] * 100 if stat['attempted_free_throws'] > 0 else 0):.3f}",
                    'ORB': stat['offensive_rebounds'],
                    'DRB': stat['defensive_rebounds'],	
                    'AST': stat['assists'],
                    'STL': stat['steals'],
                    'BLK': stat['blocks'],
                    'TOV': stat['turnovers'],
                    'PF': stat['personal_fouls'],
                    'PTS': stat['points_scored'], 
                    'GmSc': stat['game_score'], 
                    '+/-': stat['plus_minus'], 
                    'Season': season  # Add season to the CSV
                })

    print(f"Stats saved to {csv_file}")

except Exception as e:
    print(f"Error writing to CSV: {e}")
