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
        "Grambling",        # 16 seed
        "Louisville",       # 8 seed
        "Creighton",        # 9 seed
        "Michigan",         # 5 seed
        "UC San Diego",     # 12 seed
        "Texas A&M",        # 4 seed
        "Yale",             # 13 seed
        "Ole Miss",         # 6 seed
        "Virginia",         # 11 seed
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
        "Merrimack",        # 16 seed
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
        "TCU",              # 11 seed
        "Kentucky",         # 3 seed
        "Troy",             # 14 seed
        "UCLA",             # 7 seed
        "Utah State",       # 10 seed
        "Tennessee",        # 2 seed
        "Wofford",          # 15 seed
    ]
    
    midwest_seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    
    # Run predictions for each region
    print("\n=== SOUTH REGION ===")
    south_results = predictor.predict_bracket(south_teams, south_seeds)
    for result in south_results:
        print(result)
    
    print("\n=== EAST REGION ===")
    east_results = predictor.predict_bracket(east_teams, east_seeds)
    for result in east_results:
        print(result)
    
    print("\n=== WEST REGION ===")
    west_results = predictor.predict_bracket(west_teams, west_seeds)
    for result in west_results:
        print(result)
    
    print("\n=== MIDWEST REGION ===")
    midwest_results = predictor.predict_bracket(midwest_teams, midwest_seeds)
    for result in midwest_results:
        print(result)
    
    # Set the regional winners directly based on our predictions
    # We'll let the model predict these based on the updated bracket
    south_winner = None
    east_winner = None
    west_winner = None
    midwest_winner = None
    
    # Find the champion from each region
    for result in south_results:
        if "National Champion:" in result:
            parts = result.split("National Champion:")
            if len(parts) > 1:
                champion_info = parts[1].strip()
                south_winner = champion_info.split(" ")[0]
    
    for result in east_results:
        if "National Champion:" in result:
            parts = result.split("National Champion:")
            if len(parts) > 1:
                champion_info = parts[1].strip()
                east_winner = champion_info.split(" ")[0]
    
    for result in west_results:
        if "National Champion:" in result:
            parts = result.split("National Champion:")
            if len(parts) > 1:
                champion_info = parts[1].strip()
                west_winner = champion_info.split(" ")[0]
    
    for result in midwest_results:
        if "National Champion:" in result:
            parts = result.split("National Champion:")
            if len(parts) > 1:
                champion_info = parts[1].strip()
                midwest_winner = champion_info.split(" ")[0]
                
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
    for result in final_results:
        print(result)
    
    print("\n=== 2025 MARCH MADNESS COMPLETE PREDICTIONS ===")
    print("South Region Winner:", south_winner)
    print("East Region Winner:", east_winner)
    print("West Region Winner:", west_winner)
    print("Midwest Region Winner:", midwest_winner)
    
    # Find the champion
    champion = None
    for result in final_results:
        if "National Champion:" in result:
            parts = result.split("National Champion:")
            if len(parts) > 1:
                champion_info = parts[1].strip()
                champion = champion_info.split(" ")[0]
            break
    
    if champion:
        print("\n🏆 2025 NCAA TOURNAMENT CHAMPION: " + champion + " 🏆")
        
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

if __name__ == "__main__":
    main() 