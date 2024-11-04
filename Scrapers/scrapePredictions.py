import json
import pandas as pd

# Load JSON data from a file
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract relevant information into a list of dictionaries
projections = []
for item in data['data']:
    attributes = item['attributes']
    player_id = item['relationships']['new_player']['data']['id']  # Extract player ID
    projections.append({
        'id': item['id'],
        'description': attributes['description'],
        'line_score': attributes['line_score'],
        'odds_type': attributes['odds_type'],
        'projection_type': attributes['projection_type'],
        'rank': attributes['rank'],
        'refundable': attributes['refundable'],
        'start_time': attributes['start_time'],
        'stat_display_name': attributes['stat_display_name'],
        'stat_type': attributes['stat_type'],
        'status': attributes['status'],
        'updated_at': attributes['updated_at'],
        'player_id': player_id  # Add player ID to the dictionary
    })

# Create a DataFrame
df = pd.DataFrame(projections)

# Clean the DataFrame
df['start_time'] = pd.to_datetime(df['start_time'])  # Convert to datetime

# Display the cleaned DataFrame
print("All projections:")
print(df)

# Filter for 'Derrick White' description
derrick_white_df = df[df['description'] == 'Derrick White']

# Print filtered results
if not derrick_white_df.empty:
    print("\nProjections for 'Derrick White':")
    print(derrick_white_df)
else:
    print("\nNo projections found for 'Derrick White'.")

# Get player information based on player_id
if not df.empty:
    # Create a mapping of player ID to player name (You may need to get player names from another source)
    player_mapping = {
        '57075': 'Derrick White',  # Example mapping, add all relevant players here
        # Add more player mappings as necessary
    }
    
    # Add player names to the DataFrame
    df['player_name'] = df['player_id'].map(player_mapping)
    
    # Print the DataFrame with player names
    print("\nProjections with Player Names:")
    print(df[['id', 'description', 'line_score', 'odds_type', 'stat_display_name', 'player_name']])
