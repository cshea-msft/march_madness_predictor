#!/usr/bin/env python
import sys
from predictor import MarchMadnessPredictor

def main():
    print("Initializing March Madness Predictor for 2025...")
    predictor = MarchMadnessPredictor()
    
    # 2025 NCAA Tournament Bracket (based on Selection Sunday results)
    # Format: List of teams in bracket order with their seeds
    
    # South Region
    south_teams = [
        "Auburn",           # 1 seed
        "Alabama St",        # 16 seed
        "Louisville",       # 8 seed
        "Creighton",        # 9 seed
        "Michigan",         # 5 seed
        "UC San Diego",     # 12 seed
        "Texas A&M",        # 4 seed
        "Yale",             # 13 seed
        "Ole Miss",         # 6 seed
        "North Carolina",   # 11 seed
        "Iowa State",       # 3 seed
        "Lipscomb",         # 14 seed
        "Marquette",        # 7 seed
        "New Mexico",       # 10 seed
        "Michigan State",   # 2 seed
        "Bryant",           # 15 seed
    ]
    
    south_seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    
    # East Region
    east_teams = [
        "Duke",             # 1 seed
        "Mount St Mary's",  # 16 seed
        "Mississippi State",# 8 seed
        "Baylor",           # 9 seed
        "Oregon",           # 5 seed
        "Liberty",          # 12 seed
        "Arizona",          # 4 seed
        "Akron",            # 13 seed
        "BYU",              # 6 seed
        "VCU",              # 11 seed
        "Wisconsin",        # 3 seed
        "Montana",          # 14 seed
        "Saint Mary's (CA)",# 7 seed
        "Vanderbilt",       # 10 seed
        "Alabama",          # 2 seed
        "Robert Morris",    # 15 seed
    ]
    
    east_seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    
    # West Region
    west_teams = [
        "Florida",          # 1 seed
        "Norfolk State",    # 16 seed
        "UConn",            # 8 seed
        "Oklahoma",         # 9 seed
        "Memphis",          # 5 seed
        "Colorado State",   # 12 seed
        "Maryland",         # 4 seed
        "Grand Canyon",     # 13 seed
        "Missouri",         # 6 seed
        "Drake",            # 11 seed
        "Texas Tech",       # 3 seed
        "UNC Wilmington",   # 14 seed
        "Kansas",           # 7 seed
        "Arkansas",         # 10 seed
        "Saint John's (NY)",# 2 seed
        "Omaha",            # 15 seed
    ]
    
    west_seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    
    # Midwest Region
    midwest_teams = [
        "Houston",          # 1 seed
        "SIU Edwardsville", # 16 seed
        "Gonzaga",          # 8 seed
        "Georgia",          # 9 seed
        "Clemson",          # 5 seed
        "McNeese",          # 12 seed
        "Purdue",           # 4 seed
        "High Point",       # 13 seed
        "Illinois",         # 6 seed
        "Xavier",           # 11 seed
        "Kentucky",         # 3 seed
        "Troy",             # 14 seed
        "UCLA",             # 7 seed
        "Utah State",       # 10 seed
        "Tennessee",        # 2 seed
        "Wofford",          # 15 seed
    ]
    
    midwest_seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    
    # Run predictions for each region
    # South Region
    print("\n=== SOUTH REGION ===")
    south_results = predictor.predict_bracket(south_teams, south_seeds)
    
    # For each region, we need to manually show the rounds, since the predictor starts with "Sweet Sixteen"
    # for a 16-team bracket. We'll explicitly display First Round and Second Round
    print("\n--- FIRST ROUND (SOUTH REGION) ---")
    display_first_round(south_teams, south_seeds, predictor)
    
    print("\n--- SECOND ROUND (SOUTH REGION) ---")
    display_second_round(south_teams, south_seeds, predictor)
    
    print("\n--- SWEET SIXTEEN (SOUTH REGION) ---")
    display_results_for_round(south_results, "Sweet Sixteen:")
    
    print("\n--- ELITE EIGHT (SOUTH REGION) ---")
    display_results_for_round(south_results, "Elite Eight:")
    
    # East Region
    print("\n=== EAST REGION ===")
    east_results = predictor.predict_bracket(east_teams, east_seeds)
    
    print("\n--- FIRST ROUND (EAST REGION) ---")
    display_first_round(east_teams, east_seeds, predictor)
    
    print("\n--- SECOND ROUND (EAST REGION) ---")
    display_second_round(east_teams, east_seeds, predictor)
    
    print("\n--- SWEET SIXTEEN (EAST REGION) ---")
    display_results_for_round(east_results, "Sweet Sixteen:")
    
    print("\n--- ELITE EIGHT (EAST REGION) ---")
    display_results_for_round(east_results, "Elite Eight:")
    
    # West Region
    print("\n=== WEST REGION ===")
    west_results = predictor.predict_bracket(west_teams, west_seeds)
    
    print("\n--- FIRST ROUND (WEST REGION) ---")
    display_first_round(west_teams, west_seeds, predictor)
    
    print("\n--- SECOND ROUND (WEST REGION) ---")
    display_second_round(west_teams, west_seeds, predictor)
    
    print("\n--- SWEET SIXTEEN (WEST REGION) ---")
    display_results_for_round(west_results, "Sweet Sixteen:")
    
    print("\n--- ELITE EIGHT (WEST REGION) ---")
    display_results_for_round(west_results, "Elite Eight:")
    
    # Midwest Region
    print("\n=== MIDWEST REGION ===")
    midwest_results = predictor.predict_bracket(midwest_teams, midwest_seeds)
    
    print("\n--- FIRST ROUND (MIDWEST REGION) ---")
    display_first_round(midwest_teams, midwest_seeds, predictor)
    
    print("\n--- SECOND ROUND (MIDWEST REGION) ---")
    display_second_round(midwest_teams, midwest_seeds, predictor)
    
    print("\n--- SWEET SIXTEEN (MIDWEST REGION) ---")
    display_results_for_round(midwest_results, "Sweet Sixteen:")
    
    print("\n--- ELITE EIGHT (MIDWEST REGION) ---")
    display_results_for_round(midwest_results, "Elite Eight:")
    
    # Find the regional winners (Elite Eight winners)
    south_winner = find_elite_eight_winner(south_results)
    east_winner = find_elite_eight_winner(east_results)
    west_winner = find_elite_eight_winner(west_results)
    midwest_winner = find_elite_eight_winner(midwest_results)
    
    # If any regional winner is None, set a default
    if not south_winner:
        south_winner = "Auburn"
    if not east_winner:
        east_winner = "Duke"
    if not west_winner:
        west_winner = "Florida"
    if not midwest_winner:
        midwest_winner = "Houston"
    
    # Final Four
    print("\n=== FINAL FOUR ===")
    final_four_teams = [south_winner, east_winner, west_winner, midwest_winner]
    final_four_seeds = []
    
    # Find seeds for Final Four teams
    for team in final_four_teams:
        if team in south_teams:
            final_four_seeds.append(south_seeds[south_teams.index(team)])
        elif team in east_teams:
            final_four_seeds.append(east_seeds[east_teams.index(team)])
        elif team in west_teams:
            final_four_seeds.append(west_seeds[west_teams.index(team)])
        elif team in midwest_teams:
            final_four_seeds.append(midwest_seeds[midwest_teams.index(team)])
    
    # Run Final Four predictions
    final_results = predictor.predict_bracket(final_four_teams, final_four_seeds)
    
    # Display Final Four results
    print("\n--- FINAL FOUR ---")
    display_results_for_round(final_results, "Final Four:")
    
    print("\n--- NATIONAL CHAMPIONSHIP ---")
    display_results_for_round(final_results, "Championship Game:")
    
    print("\n=== 2025 MARCH MADNESS COMPLETE PREDICTIONS ===")
    print("South Region Winner:", south_winner)
    print("East Region Winner:", east_winner)
    print("West Region Winner:", west_winner)
    print("Midwest Region Winner:", midwest_winner)
    
    # Find the champion
    champion = find_national_champion(final_results)
    
    if champion:
        print("\n=== NATIONAL CHAMPIONSHIP ===")
        print("🏆 2025 NCAA TOURNAMENT CHAMPION: " + champion + " 🏆")
        
        # Get the champion's seed
        champion_seed = None
        if champion in south_teams:
            champion_seed = south_seeds[south_teams.index(champion)]
        elif champion in east_teams:
            champion_seed = east_seeds[east_teams.index(champion)]
        elif champion in west_teams:
            champion_seed = west_seeds[west_teams.index(champion)]
        elif champion in midwest_teams:
            champion_seed = midwest_seeds[midwest_teams.index(champion)]
            
        print(f"\nSeed: #{champion_seed}")
        print(f"Region: {['South', 'East', 'West', 'Midwest'][final_four_teams.index(champion)]}")
        
        # Get team stats
        team_stats = predictor.team_stats.get(champion, {})
        if team_stats:
            print(f"\nKey Statistics:")
            print(f"Win Rate: {team_stats.get('win_rate', 0.0)}")
            print(f"Points per Game: {team_stats.get('points_per_game', 0.0)}")
            print(f"Defensive Rating: {team_stats.get('defensive_rating', 0.0)}")
            print(f"Strength of Schedule: {team_stats.get('sos', 0.0)}")
            print(f"Simple Rating System: {team_stats.get('srs', 0.0)}")

def display_first_round(teams, seeds, predictor):
    """Display the First Round matchups and results"""
    print("First Round:")
    for i in range(0, len(teams), 2):
        team1 = teams[i]
        team2 = teams[i+1]
        seed1 = seeds[i]
        seed2 = seeds[i+1]
        
        winner, confidence = predictor.predict_game(team1, team2, seed1, seed2, "First Round")
        
        # Format the result
        seed1_display = f"({seed1})" if seed1 is not None else ""
        seed2_display = f"({seed2})" if seed2 is not None else ""
        result_str = f"{team1} {seed1_display} vs {team2} {seed2_display}: {winner} wins ({confidence:.1%} confidence)"
        
        # Add any special notes about the matchup
        matchup_note = predictor._get_matchup_note(team1, team2, seed1, seed2, "First Round")
        if matchup_note:
            result_str += f" - {matchup_note}"
        
        print(result_str)

def display_second_round(teams, seeds, predictor):
    """Display the Second Round matchups and results based on First Round winners"""
    print("Second Round:")
    second_round_teams = []
    second_round_seeds = []
    
    # Determine winners from First Round
    for i in range(0, len(teams), 2):
        team1 = teams[i]
        team2 = teams[i+1]
        seed1 = seeds[i]
        seed2 = seeds[i+1]
        
        winner, _ = predictor.predict_game(team1, team2, seed1, seed2, "First Round")
        second_round_teams.append(winner)
        second_round_seeds.append(seed1 if winner == team1 else seed2)
    
    # Now predict Second Round matchups
    for i in range(0, len(second_round_teams), 2):
        team1 = second_round_teams[i]
        team2 = second_round_teams[i+1]
        seed1 = second_round_seeds[i]
        seed2 = second_round_seeds[i+1]
        
        winner, confidence = predictor.predict_game(team1, team2, seed1, seed2, "Second Round")
        
        # Format the result
        seed1_display = f"({seed1})" if seed1 is not None else ""
        seed2_display = f"({seed2})" if seed2 is not None else ""
        result_str = f"{team1} {seed1_display} vs {team2} {seed2_display}: {winner} wins ({confidence:.1%} confidence)"
        
        # Add any special notes about the matchup
        matchup_note = predictor._get_matchup_note(team1, team2, seed1, seed2, "Second Round")
        if matchup_note:
            result_str += f" - {matchup_note}"
        
        print(result_str)

def display_results_for_round(results, round_header):
    """Display results for a specific round"""
    # Find the round's start index
    start_idx = -1
    for i, result in enumerate(results):
        if round_header in result:
            start_idx = i
            break
    
    if start_idx == -1:
        print(f"No results found for {round_header}")
        return
    
    # Find the next round's start index
    next_start_idx = len(results)
    round_headers = ["First Round:", "Second Round:", "Sweet Sixteen:", "Elite Eight:", "Final Four:", "Championship Game:"]
    for header in round_headers:
        if header == round_header:
            continue
        
        for i in range(start_idx + 1, len(results)):
            if header in results[i]:
                next_start_idx = i
                break
        
        if next_start_idx < len(results):
            break
    
    # Print all results for this round
    print(results[start_idx])  # Print the round header
    
    # Count the number of games to show based on the round
    games_to_show = 0
    if "Second Round:" in round_header:
        games_to_show = 4
    elif "Sweet Sixteen:" in round_header:
        games_to_show = 2
    elif "Elite Eight:" in round_header:
        games_to_show = 1
    else:
        games_to_show = float('inf')  # Show all games for other rounds
    
    # Print the appropriate number of games
    game_count = 0
    for i in range(start_idx + 1, next_start_idx):
        if "National Champion:" not in results[i] and "wins" in results[i]:
            print(results[i])
            game_count += 1
            if game_count >= games_to_show:
                break

def find_elite_eight_winner(results):
    """Find the winner of the Elite Eight game (regional champion)"""
    elite_eight_section = False
    for i, result in enumerate(results):
        if "Elite Eight:" in result:
            elite_eight_section = True
            # Look for the last game result in the Elite Eight section
            for j in range(i+1, len(results)):
                if "Final Four:" in results[j]:
                    break
                if "wins" in results[j]:
                    parts = results[j].split(":")
                    if len(parts) > 1:
                        winner_part = parts[1].split("wins")[0].strip()
                        return winner_part
    return None

def find_national_champion(results):
    """Find the national champion from the Final Four results"""
    for result in results:
        if "National Champion:" in result:
            parts = result.split("National Champion:")
            if len(parts) > 1:
                champion_info = parts[1].strip()
                return champion_info.split(" ")[0]
    return None

if __name__ == "__main__":
    main() 