#!/usr/bin/env python
import argparse
import csv
import logging
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Base URL for NCAA tournament data
BASE_URL = "https://www.sports-reference.com/cbb/postseason/"

def collect_tournament_data(year: int) -> List[Dict[str, Any]]:
    """
    Collect NCAA tournament data for a specific year.
    
    Args:
        year: The year to collect data for
        
    Returns:
        List of dictionaries with game data
    """
    logger.info(f"Collecting NCAA tournament data for {year}")
    
    url = f"{BASE_URL}{year}-ncaa.html"
    
    # In a real implementation, this would scrape the actual website
    # For this example, we'll generate synthetic data
    
    # Wait to simulate network request
    time.sleep(0.5)
    
    # Generate synthetic tournament data
    seeds = list(range(1, 17))
    regions = ["East", "West", "South", "Midwest"]
    
    # Teams with typical seeding
    teams_by_seed = {
        1: ["Gonzaga", "Kansas", "Baylor", "Arizona"],
        2: ["Duke", "Kentucky", "Auburn", "Villanova"],
        3: ["Tennessee", "Texas Tech", "Purdue", "Wisconsin"],
        4: ["UCLA", "Arkansas", "Illinois", "Providence"],
        5: ["Houston", "Saint Mary's", "Iowa", "Connecticut"],
        6: ["Alabama", "Texas", "LSU", "Colorado State"],
        7: ["Michigan State", "Ohio State", "USC", "Murray State"],
        8: ["North Carolina", "Boise State", "San Diego State", "Seton Hall"],
        9: ["Marquette", "Memphis", "Creighton", "TCU"],
        10: ["Miami", "Davidson", "Loyola Chicago", "San Francisco"],
        11: ["Notre Dame", "Virginia Tech", "Iowa State", "Michigan"],
        12: ["New Mexico State", "Richmond", "UAB", "Indiana"],
        13: ["Vermont", "South Dakota State", "Chattanooga", "Akron"],
        14: ["Montana State", "Yale", "Colgate", "Longwood"],
        15: ["Saint Peter's", "CSU Fullerton", "Jacksonville State", "Delaware"],
        16: ["Georgia State", "Norfolk State", "Wright State", "Bryant"]
    }
    
    # Adjust team names slightly for historical variation
    for seed in teams_by_seed:
        teams_by_seed[seed] = [f"{team} ({year})" for team in teams_by_seed[seed]]
    
    # Generate tournament games with realistic outcomes
    games = []
    
    # First round matchups
    for region_idx, region in enumerate(regions):
        for i in range(8):
            higher_seed = i + 1
            lower_seed = 17 - higher_seed
            
            team1 = teams_by_seed[higher_seed][region_idx]
            team2 = teams_by_seed[lower_seed][region_idx]
            
            # Higher seeds win more often, especially with bigger seed gaps
            upset_prob = 0.025 * (17 - higher_seed)
            winner = team2 if random.random() < upset_prob else team1
            
            games.append({
                "year": year,
                "round": "First Round",
                "region": region,
                "team1": team1,
                "team2": team2,
                "seed1": higher_seed,
                "seed2": lower_seed,
                "winner": winner,
                "team1_win": winner == team1
            })
    
    # In a real implementation, we would generate all rounds
    # For simplicity, we're only generating the first round here
    
    return games

def collect_data(start_year: int = 2000, end_year: int = 2024, output_file: str = "historical_march_madness.csv") -> None:
    """
    Collect historical NCAA tournament data for multiple years.
    
    Args:
        start_year: First year to collect data for
        end_year: Last year to collect data for
        output_file: Path to save the data to
    """
    logger.info(f"Collecting historical NCAA tournament data from {start_year} to {end_year}")
    
    all_games = []
    for year in range(start_year, end_year + 1):
        # Skip 2020 since tournament was canceled due to COVID-19
        if year == 2020:
            logger.info(f"Skipping 2020 (tournament canceled)")
            continue
            
        year_games = collect_tournament_data(year)
        all_games.extend(year_games)
        
        # Be nice to the server by adding delay between years
        if year < end_year:
            time.sleep(1)
    
    # Save collected data to CSV
    logger.info(f"Saving {len(all_games)} games to {output_file}")
    
    with open(output_file, 'w', newline='') as f:
        if all_games:
            fieldnames = all_games[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_games)
    
    logger.info("Data collection complete!")

def main():
    parser = argparse.ArgumentParser(description="Collect historical NCAA tournament data")
    parser.add_argument("--start-year", type=int, default=2010,
                      help="First year to collect data for")
    parser.add_argument("--end-year", type=int, default=2024,
                      help="Last year to collect data for")
    parser.add_argument("--output", type=str, default="historical_march_madness.csv",
                      help="Output file path")
    args = parser.parse_args()
    
    collect_data(args.start_year, args.end_year, args.output)

if __name__ == "__main__":
    main() 