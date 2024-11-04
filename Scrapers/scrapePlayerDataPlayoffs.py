from basketball_reference_web_scraper import client
import csv
from datetime import datetime
from basketball_reference_web_scraper.data import OutputType

# Define the CSV file and headers
playerid = 'westbru01'
start_year = 2023
csv_file = 'PlayerStats/westbru01.csv'
csv_columns = [
    'Rk', 'G', 'Date', 'Tm', 'Opp', 'MP', 
    'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 
    'PTS', 'GmSc', '+/-', 'season', 'playoffs'
]

# Team abbreviations mapping
team_abbreviations = {
    # (mapping as you provided)
}

# Define the seasons and their corresponding date ranges
Seasons = {
    # (season mapping as you provided)
}

# Save stats to CSV
try:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()

        # Loop through all seasons
        for season_year in range(start_year, 2026):  # Adjust the range as needed
            try:
                # Scrape regular season stats for each season
                player_stats = client.regular_season_player_box_scores(
                    player_identifier=playerid,
                    season_end_year=season_year
                )

                # Write regular season stats to CSV
                for game_number, stat in enumerate(player_stats, start=1):
                    game_date = stat['date']
                    game_date_str = game_date.strftime("%Y-%m-%d")  # Format the date for CSV

                    season = None
                    for season_key, (start_date, end_date) in Seasons.items():
                        if start_date <= game_date_str <= end_date:
                            season = season_key
                            break
                    
                    if season is None:  # If no valid season found, skip to next
                        print(f"Warning: No valid season found for game date {game_date_str}. Skipping game.")
                        continue

                    opponent = stat['opponent'].value
                    opponent_abbreviation = team_abbreviations.get(opponent, opponent)
                    team_abbreviation = team_abbreviations.get(stat['team'].value, stat['team'].value)

                    writer.writerow({
                        'Rk': game_number,
                        'G': game_number,
                        'Date': game_date_str,
                        'Tm': team_abbreviation,
                        'Opp': opponent_abbreviation,
                        'MP': f"{stat['seconds_played'] // 60}:{stat['seconds_played'] % 60:02}",
                        'FG': stat['made_field_goals'],
                        'FGA': stat['attempted_field_goals'],
                        'FG%': f"{(stat['made_field_goals'] / stat['attempted_field_goals'] * 100 if stat['attempted_field_goals'] > 0 else 0):.3f}",
                        '3P': stat['made_three_point_field_goals'],
                        '3PA': stat['attempted_three_point_field_goals'],
                        '3P%': f"{(stat['made_three_point_field_goals'] / stat['attempted_three_point_field_goals'] * 100 if stat['attempted_three_point_field_goals'] > 0 else 0):.3f}",
                        'FT': stat['made_free_throws'],
                        'FTA': stat['attempted_free_throws'],
                        'FT%': f"{(stat['made_free_throws'] / stat['attempted_free_throws'] * 100 if stat['attempted_free_throws'] > 0 else 0):.3f}",
                        'PTS': stat['points_scored'],
                        'GmSc': stat['game_score'],
                        '+/-': stat['plus_minus'],
                        'season': season,
                        'playoffs': 'No'
                    })

                # Scrape playoff stats for each season
                try:
                    playoff_stats = client.playoff_player_box_scores(
                        player_identifier=playerid,
                        season_end_year=season_year
                    )

                    # Write playoff stats to CSV
                    for game_number, stat in enumerate(playoff_stats, start=1):
                        game_date = stat['date']
                        game_date_str = game_date.strftime("%Y-%m-%d")  # Format the date for CSV

                        opponent = stat['opponent'].value
                        opponent_abbreviation = team_abbreviations.get(opponent, opponent)
                        team_abbreviation = team_abbreviations.get(stat['team'].value, stat['team'].value)

                        writer.writerow({
                            'Rk': game_number,
                            'G': game_number,
                            'Date': game_date_str,
                            'Tm': team_abbreviation,
                            'Opp': opponent_abbreviation,
                            'MP': f"{stat['seconds_played'] // 60}:{stat['seconds_played'] % 60:02}",
                            'FG': stat['made_field_goals'],
                            'FGA': stat['attempted_field_goals'],
                            'FG%': f"{(stat['made_field_goals'] / stat['attempted_field_goals'] * 100 if stat['attempted_field_goals'] > 0 else 0):.3f}",
                            '3P': stat['made_three_point_field_goals'],
                            '3PA': stat['attempted_three_point_field_goals'],
                            '3P%': f"{(stat['made_three_point_field_goals'] / stat['attempted_three_point_field_goals'] * 100 if stat['attempted_three_point_field_goals'] > 0 else 0):.3f}",
                            'FT': stat['made_free_throws'],
                            'FTA': stat['attempted_free_throws'],
                            'FT%': f"{(stat['made_free_throws'] / stat['attempted_free_throws'] * 100 if stat['attempted_free_throws'] > 0 else 0):.3f}",
                            'PTS': stat['points_scored'],
                            'GmSc': stat['game_score'],
                            '+/-': stat['plus_minus'],
                            'season': season,
                            'playoffs': 'Yes'
                        })

                except Exception as e:
                    print(f"An error occurred while scraping playoff stats for season {season_year}: {e}")

            except Exception as e:
                print(f"An error occurred while processing season {season_year}: {e}")

except Exception as e:
    print(f"An error occurred while opening the CSV file: {e}")
