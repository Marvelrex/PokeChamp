# Pokemon Battle Log Analyzer
# This script analyzes Pokemon Showdown battle logs saved as HTML files
# to determine winners, count wins, and calculate win rates for each player.

# To use this script:
# 1. Make sure you have BeautifulSoup installed:
#    pip install beautifulsoup4
# 2. Place this script in the same folder as your HTML battle log files.
# 3. Run the script from your terminal:
#    python analyze_battles.py

import os
import re
from bs4 import BeautifulSoup


def get_base_player_name(name):
    """Removes trailing digits from a player name to group them."""
    return re.sub(r'\d+$', '', name)


def analyze_battle_logs(folder_path, accuracy):
    """
    Analyzes all HTML battle log files in a given folder to determine
    winners and calculate player statistics.

    Args:
        folder_path (str): The path to the folder containing the HTML files.
    """
    player_stats = {}

    print("--- Battle Analysis Results ---\n")

    # Find all HTML files in the specified directory
    html_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]

    if not html_files:
        print(f"No HTML files found in the directory: {folder_path}")
        return

    for filename in html_files:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        log_data_script = soup.find('script', class_='battle-log-data')

        if not log_data_script:
            print(f"Could not find battle log in: {filename}")
            continue

        log_content = log_data_script.string
        lines = log_content.strip().split('\n')

        players = {}
        winner = None

        # Extract player information
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) > 2:
                if parts[1] == 'player':
                    player_id = parts[2]
                    player_name = parts[3]
                    players[player_id] = player_name
                    # Initialize stats for new players
                    base_player_name = get_base_player_name(player_name)
                    if base_player_name not in player_stats:
                        player_stats[base_player_name] = {'wins': 0, 'games': 0}

                # Check for a win condition in the log
                elif parts[1] == 'win':
                    winner_name = parts[2]
                    # Sometimes the winner name might have extra characters
                    # We find the player name that is a substring of the winner message
                    for name in players.values():
                        if name.lower() in winner_name.lower():
                            winner = get_base_player_name(name)
                            break

        # If there's no explicit win message, determine winner by fainted Pokemon
        if not winner:
            team_sizes = {}
            faint_counts = {'p1': 0, 'p2': 0}
            for line in lines:
                parts = line.strip().split('|')
                if len(parts) > 2:
                    if parts[1] == 'teamsize':
                        team_sizes[parts[2]] = int(parts[3])
                    elif parts[1] == 'faint' and parts[2].startswith(('p1', 'p2')):
                        # e.g |faint|p1a: Dragapult
                        player_id = parts[2][:2]
                        if player_id in faint_counts:
                            faint_counts[player_id] += 1

            if 'p1' in team_sizes and 'p2' in team_sizes:
                winner_original = None
                if faint_counts['p2'] >= team_sizes['p2'] and faint_counts['p1'] < team_sizes['p1']:
                    winner_original = players.get('p1')
                elif faint_counts['p1'] >= team_sizes['p1'] and faint_counts['p2'] < team_sizes['p2']:
                    winner_original = players.get('p2')

                if winner_original:
                    winner = get_base_player_name(winner_original)

        # Update game counts for both players if they were identified
        if players:
            player_list = list(players.values())
            # Ensure we have two distinct players
            if len(player_list) == 2 and player_list[0] and player_list[1]:
                base_player_1 = get_base_player_name(player_list[0])
                base_player_2 = get_base_player_name(player_list[1])
                if base_player_1 in player_stats:
                    player_stats[base_player_1]['games'] += 1
                if base_player_2 in player_stats:
                    player_stats[base_player_2]['games'] += 1

        if winner:
            if winner in player_stats:
                player_stats[winner]['wins'] += 1
        else:
            print(f"File: {filename} -> No clear winner (game might have been stopped or ended in a tie).")

    print(f"\n--- {accuracy} Player Win Rate Summary ---")

    if not player_stats:
        print("No player data to summarize.")
        return

    # Sort players by win rate for better readability
    sorted_players = sorted(player_stats.items(),
                            key=lambda item: (item[1]['wins'] / item[1]['games']) if item[1]['games'] > 0 else 0,
                            reverse=True)

    for player_name, stats in sorted_players:
        wins = stats['wins']
        games = stats['games']
        if games > 0:
            win_rate = (wins / games) * 100
            print(f"{player_name}: {wins} Wins in {games} Games (Win Rate: {win_rate:.2f}%)")
        else:
            print(f"{player_name}: 0 Games Played")



if __name__ == "__main__":
    # The script will analyze files in the same directory it is run from.
    current_folder = os.getcwd()
    analyze_battle_logs("/home/jialinlabvm/Documents/pokechamp/battle_log/WinRate_Calculator/High",'high')
    analyze_battle_logs('/home/jialinlabvm/Documents/pokechamp/battle_log/WinRate_Calculator/Low','low')