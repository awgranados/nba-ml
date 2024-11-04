from basketball_reference_web_scraper import client
import csv
import os
import time  # Import time module for adding delays

# Define the CSV headers
csv_columns = [
    'Rk', 'G', 'Date', 'Tm', 'Opp', 'MP', 
    'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 
    'PTS', 'GmSc', '+/-'
]

# Team abbreviations mapping
team_abbreviations = {
    'ATLANTA HAWKS': 'ATL', 'BOSTON CELTICS': 'BOS', 'BROOKLYN NETS': 'BKN',
    'CHARLOTTE HORNETS': 'CHA', 'CHICAGO BULLS': 'CHI', 'CLEVELAND CAVALIERS': 'CLE',
    'DALLAS MAVERICKS': 'DAL', 'DENVER NUGGETS': 'DEN', 'DETROIT PISTONS': 'DET',
    'GOLDEN STATE WARRIORS': 'GSW', 'HOUSTON ROCKETS': 'HOU', 'INDIANA PACERS': 'IND',
    'LOS ANGELES CLIPPERS': 'LAC', 'LOS ANGELES LAKERS': 'LAL', 'MEMPHIS GRIZZLIES': 'MEM',
    'MIAMI HEAT': 'MIA', 'MILWAUKEE BUCKS': 'MIL', 'MINNESOTA TIMBERWOLVES': 'MIN',
    'NEW ORLEANS PELICANS': 'NOP', 'NEW YORK KNICKS': 'NYK', 'OKLAHOMA CITY THUNDER': 'OKC',
    'ORLANDO MAGIC': 'ORL', 'PHILADELPHIA 76ERS': 'PHI', 'PHOENIX SUNS': 'PHX',
    'PORTLAND TRAIL BLAZERS': 'POR', 'SACRAMENTO KINGS': 'SAC', 'SAN ANTONIO SPURS': 'SAS',
    'TORONTO RAPTORS': 'TOR', 'UTAH JAZZ': 'UTA', 'WASHINGTON WIZARDS': 'WAS'
}

# Read players from input CSV and scrape stats
def scrape_player_stats(input_csv):
    try:
        with open(input_csv, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_id = row['playerID']
                start_year = int(row['startYear'])
                csv_file = f'PlayerStats/{player_id}.csv'

                # Make sure directory exists
                os.makedirs(os.path.dirname(csv_file), exist_ok=True)

                # Save stats to CSV
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=csv_columns)
                    writer.writeheader()

                    for season_year in range(start_year, 2025):  # Adjust range as needed
                        player_stats = client.regular_season_player_box_scores(
                            player_identifier=player_id,
                            season_end_year=season_year
                        )

                        for game_number, stat in enumerate(player_stats, start=1):
                            game_date = stat['date']
                            opponent_abbreviation = team_abbreviations.get(stat['opponent'].value, stat['opponent'].value)
                            team_abbreviation = team_abbreviations.get(stat['team'].value, stat['team'].value)

                            # Write stats row
                            writer.writerow({
                                'Rk': game_number,
                                'G': game_number,
                                'Date': game_date.strftime("%Y-%m-%d"),
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
                            })

                        # Add a delay between requests
                        time.sleep(5)  # Adjust the delay (in seconds) as needed

                print(f"Stats saved for {player_id} to {csv_file}")

    except Exception as e:
        print(f"Error processing {input_csv}: {e}")

# Run the scraper
input_csv = 'MIA25.csv'  # Replace with the path to your input CSV
scrape_player_stats(input_csv)
