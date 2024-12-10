# Install necessary packages
!pip install shap
!pip install optuna

# Import required libraries
import pandas as pd
import pickle as p
from tqdm import tqdm
from scipy.spatial import distance
import math
import numpy as np
from typing_extensions import final

# Base path for the dataset
base_path = "/content/drive/MyDrive/nfl-big-data-bowl-2024/"

# Load dataset files
dfgames = pd.read_csv(base_path + "games.csv")
dfplayers = pd.read_csv(base_path + "players.csv")
dfplays = pd.read_csv(base_path + "plays.csv")
dftackles = pd.read_csv(base_path + "tackles.csv")
df_track = pd.read_parquet(base_path + "df_tracking.parquet")

# Display columns of each dataset
print(dfgames.columns)
print(dfplayers.columns)
print(dfplays.columns)
print(dftackles.columns)
print(df_track.columns)

# Create a dictionary of tracking data for each game
games = {}
for gameId in dfgames['gameId']:
    games[gameId] = df_track[df_track['gameId'] == gameId]

# Function to find the closest point to a given center point
def closest_node(center_point, surrounding_points):
    """
    Finds the closest point to the given center point from a list of surrounding points.
    """
    closest_index = distance.cdist([center_point], surrounding_points).argmin()
    return surrounding_points[closest_index]

# Function to calculate the distance between two points
def calculateDistance(x1, y1, x2, y2):
    """
    Calculates Euclidean distance between two points (x1, y1) and (x2, y2).
    """
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

# Function to calculate catch separation
def addCatchSeparation(indf, dfplays=dfplays, dfgames=dfgames):
    """
    Calculates the separation between the catching receiver and the closest defender during a pass play.
    """
    game_id = indf["gameId"].values[0]
    play_id = indf["playId"].values[0]

    # Retrieve ball carrier (catching receiver) information
    carrier_id = dfplays[(dfplays["gameId"] == game_id) & (dfplays["playId"] == play_id)]["ballCarrierId"].values[0]
    df_ball_tracking = indf[(indf["nflId"] == carrier_id)]

    # Retrieve defending team and their players' positions
    defensive_team = dfplays[(dfplays["gameId"] == game_id) & (dfplays["playId"] == play_id)]["defensiveTeam"].values[0]
    df_team_data = indf[(indf["frameId"] == df_ball_tracking["frameId"].values[0]) & (indf['club'] == defensive_team)]

    # Get positions of defensive players and the receiver's position
    defensive_positions = df_team_data[['x', 'y']].values
    receiver_x, receiver_y = df_ball_tracking['x'].values[0], df_ball_tracking['y'].values[0]

    # Find closest defender to the ball carrier
    closest_defender_coords = closest_node((receiver_x, receiver_y), defensive_positions)
    closest_defender = df_team_data[(df_team_data['x'] == closest_defender_coords[0]) &
                                    (df_team_data['y'] == closest_defender_coords[1])]

    # Calculate the distance (catch separation) between the receiver and the closest defender
    catch_separation = calculateDistance(receiver_x, receiver_y,
                                         closest_defender['x'].values[0], closest_defender['y'].values[0])

    # Determine offense and defense teams
    offense_team = df_ball_tracking['club'].values[0]
    home_team = dfgames[dfgames['gameId'] == game_id]['homeTeamAbbr'].values[0]
    away_team = dfgames[dfgames['gameId'] == game_id]['visitorTeamAbbr'].values[0]

    # Assign 'home' or 'away' labels to offense and defense
    offense_label = "home" if offense_team == home_team else "away"
    defense_label = "away" if offense_team == home_team else "home"

    # Create the output DataFrame with consistent column names
    dfcatch_wSeparation = pd.DataFrame({
        'gameId': [game_id],
        'playId': [play_id],
        'catchSeparation': [catch_separation],
        'catchingReceiver': [df_ball_tracking['displayName'].values[0]],
        'closestCorner': [closest_defender['displayName'].values[0]],
        'receiver x': [receiver_x],
        'receiver y': [receiver_y],
        'offense': [offense_label],
        'defense': [defense_label]
    })

    return dfcatch_wSeparation

# Function to process a specific play in a game
def process_play(play, dfgame):
    """
    Processes a single play to calculate catch separation metrics.
    """
    try:
        playdf = dfgame[dfgame['playId'] == play]
        catch_frame = playdf[playdf['event'] == 'pass_outcome_caught']['frameId'].values[0]
        indf = playdf[playdf['frameId'] == catch_frame]
        return addCatchSeparation(indf)
    except Exception as e:
        print(f"Failed to calculate catch separation for play {play} in game {dfgame['gameId'].iloc[0]}")
        print(e)
        return None

# Function to process all plays in a game
def process_game(game_id, games):
    """
    Processes all plays in a single game to compute catch separation metrics for each play.
    """
    dfgame = games[game_id]
    play_ids = dfgame[dfgame['event'] == 'pass_outcome_caught']['playId'].unique()

    game_df = pd.DataFrame()
    for play in play_ids:
        play_df = process_play(play, dfgame)
        if play_df is not None:
            game_df = pd.concat([game_df, play_df], ignore_index=True)

    return game_df[['gameId', 'playId', 'catchingReceiver', 'closestCorner',
                    'receiver x', 'receiver y', 'catchSeparation',
                    'offense', 'defense']].drop_duplicates()

# Function to process all games and compile data
def compile_games_data(games):
    """
    Processes all games in the dataset to compile a consolidated DataFrame of catch separation metrics.
    """
    main_df = pd.DataFrame()
    for main_counter, game_id in enumerate(games):
        game_df = process_game(game_id, games)
        main_df = pd.concat([main_df, game_df], ignore_index=True)

    return main_df

# Compile data for all games
main_df = compile_games_data(games)

# The resulting main_df contains all the engineered features.

df = df_track.dropna(subset=['event']).reset_index(drop=True)

def find_next_event(group):
    """
    Identifies the next significant event after the last 'pass_outcome_caught' event 
    in a group and calculates the distance between the events.

    Parameters:
    group (DataFrame): Subset of data grouped by 'gameId' and 'playId'.

    Returns:
    tuple: A tuple containing gameId, playId, next event after the catch, and the distance.
    """
    global count
    group = group.reset_index(drop=True)

    # Locate the index of the last 'pass_outcome_caught' event in the group
    catch_idx = group[group['event'] == 'pass_outcome_caught'].index
    if len(catch_idx) == 0:
        return 0, 0, "Not applicable", 0  # No 'pass_outcome_caught' event found

    # Get the index of the last 'pass_outcome_caught'
    last_catch_idx = catch_idx[-1]
    next_event_idx = last_catch_idx + 1

    # Retrieve initial x-coordinate and other relevant details
    initial_dis = group.loc[last_catch_idx, 'x']
    playdirection = group.loc[last_catch_idx, 'playDirection']
    gameId = group.loc[last_catch_idx, 'gameId']
    playId = group.loc[last_catch_idx, 'playId']

    # Skip consecutive 'first_contact' events to find the next relevant event
    while next_event_idx < len(group) and group.loc[next_event_idx, 'event'] == 'first_contact':
        next_event_idx += 1  # Move to the next event index

    # Check if a valid next event exists
    if next_event_idx < len(group):
        next_event = group.loc[next_event_idx, 'event']
        distance = 0
        # Calculate distance if the event is 'tackle' or 'out_of_bounds'
        if next_event == 'tackle' or next_event == 'out_of_bounds':
            final_dis = group.loc[next_event_idx, 'x']
            if initial_dis > final_dis and playdirection == 'left':
                distance = initial_dis - final_dis
            elif final_dis > initial_dis and playdirection == 'right':
                distance = final_dis - initial_dis
        return gameId, playId, next_event, distance
    else:
        return gameId, playId, "End of group", 0  # No valid next event found

# Apply the function across groups and unpack the results into a DataFrame
result = df.groupby(['gameId', 'playId']).apply(find_next_event).reset_index(level=[0, 1], drop=True)

# Convert the results into a structured DataFrame with separate columns
result_df = pd.DataFrame(result.tolist(), columns=['gameId', 'playId', 'next_event_after_catch', 'distance'])

# Display the resulting DataFrame
print(result_df)

# Determine pass outcome based on the next event and distance
# Successful play is defined as either:
# - A non-tackle/non-out_of_bounds event
# - A tackle/out_of_bounds event with a distance > 5 yards

result_df['pass_outcome'] = np.where(
    ~result_df['next_event_after_catch'].isin(['tackle', 'out_of_bounds']),
    1,  # Successful play
    np.where(result_df['distance'] > 5, 1, 0)  # Successful if distance > 5
)

# Filter out rows with 'Not applicable' next events
result_df2 = result_df[result_df['next_event_after_catch'] != 'Not applicable']

# Extract relevant columns for the final output
final_df = result_df2[['gameId', 'playId', 'pass_outcome']]

# Merge the calculated pass outcome with the main DataFrame
main_df4 = pd.merge(main_df3, final_df, how='left', on=['gameId', 'playId'])

# Uncomment the following line to save the result to a CSV file
# main_df4.to_csv('/content/drive/My Drive/nfl-big-data-bowl-2024/main_df5.csv', index=False)

# Display the merged DataFrame
main_df4

# Calculate success probability for each catching receiver
main_df = main_df4.copy(deep=True)
success_prob_df = (
    main_df.groupby('catchingReceiver')  # Group by catching receiver
    .apply(lambda x: 0 if len(x) < 20 else (x['pass_outcome'] == 1).sum() / len(x))  # Success probability
    .reset_index(name='success_probability')  # Reset index and rename column
)

# Filter out players with non-zero success probability
sorted_success_df = success_prob_df[success_prob_df['success_probability'] != 0]
sorted_success_df = sorted_success_df.reset_index(drop=True)

# Sort players by their success probability in descending order
sorted_success_df = sorted_success_df.sort_values(by='success_probability', ascending=False)

sorted_success_df

