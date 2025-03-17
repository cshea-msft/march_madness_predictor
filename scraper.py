import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from datetime import datetime
import time
from typing import Dict, Any
import logging

class CBBStatsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = "https://www.sports-reference.com/cbb/seasons/men/2024-ratings.html"
        self.stats_cache_file = "team_stats_cache.json"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _fetch_page(self, url: str) -> str:
        """Fetch webpage content with error handling and rate limiting."""
        try:
            time.sleep(3)  # Rate limiting to be respectful to the server
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return ""

    def _parse_team_stats(self, html: str) -> Dict[str, Any]:
        """Parse team statistics from the HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        stats = {}
        
        try:
            # Basic team info
            stats['team_name'] = soup.select_one('h1[itemprop="name"]').text.strip()
            
            # Find the main stats table
            stats_table = soup.find('table', {'id': 'season-stats'})
            if stats_table:
                current_season = stats_table.find('tr', {'class': 'current'})
                if current_season:
                    stats.update({
                        'wins': int(current_season.find('td', {'data-stat': 'wins'}).text),
                        'losses': int(current_season.find('td', {'data-stat': 'losses'}).text),
                        'win_rate': float(current_season.find('td', {'data-stat': 'win_loss_pct'}).text),
                        'points_per_game': float(current_season.find('td', {'data-stat': 'pts_per_g'}).text),
                        'points_allowed': float(current_season.find('td', {'data-stat': 'opp_pts_per_g'}).text),
                        'srs': float(current_season.find('td', {'data-stat': 'srs'}).text),  # Simple Rating System
                        'sos': float(current_season.find('td', {'data-stat': 'sos'}).text),  # Strength of Schedule
                    })
            
            # Advanced stats
            advanced_table = soup.find('table', {'id': 'advanced-stats'})
            if advanced_table:
                current_season = advanced_table.find('tr', {'class': 'current'})
                if current_season:
                    stats.update({
                        'pace': float(current_season.find('td', {'data-stat': 'pace'}).text),
                        'offensive_rating': float(current_season.find('td', {'data-stat': 'off_rtg'}).text),
                        'defensive_rating': float(current_season.find('td', {'data-stat': 'def_rtg'}).text),
                    })
            
            return stats
        except Exception as e:
            self.logger.error(f"Error parsing stats for {stats.get('team_name', 'unknown team')}: {str(e)}")
            return {}

    def scrape_all_teams(self) -> Dict[str, Dict[str, Any]]:
        """Scrape statistics for all Division I teams."""
        all_teams_stats = {}
        
        try:
            # Get the main ratings page
            main_page = self._fetch_page(self.base_url)
            if not main_page:
                self.logger.error("Failed to fetch main page")
                return {}
                
            soup = BeautifulSoup(main_page, 'lxml')
            
            # Find the ratings table
            ratings_table = soup.find('table', {'id': 'ratings'})
            if not ratings_table:
                self.logger.error("No ratings table found")
                return {}
            
            # Process each team row
            team_rows = ratings_table.select('tbody tr:not(.thead)')  # Exclude header rows
            valid_rows = [row for row in team_rows if not row.get('class', [''])[0] in ['thead', 'spacer']]
            self.logger.info(f"Found {len(valid_rows)} teams")
            
            for row in valid_rows:
                try:
                    # Check if row has the required cells
                    name_cell = row.find('td', {'data-stat': 'school_name'})
                    if not name_cell or not name_cell.text.strip():
                        continue
                        
                    team_name = name_cell.text.strip()
                    
                    # Helper function to safely extract numeric values
                    def safe_extract(row, stat_name, default=0.0):
                        cell = row.find('td', {'data-stat': stat_name})
                        if cell and cell.text.strip():
                            try:
                                return float(cell.text.strip())
                            except (ValueError, TypeError):
                                return default
                        return default
                    
                    # Extract statistics with safe defaults
                    stats = {
                        'team_name': team_name,
                        'wins': int(safe_extract(row, 'wins', 0)),
                        'losses': int(safe_extract(row, 'losses', 0)),
                        'win_rate': safe_extract(row, 'win_loss_pct', 0.0),
                        'srs': safe_extract(row, 'srs', 0.0),
                        'sos': safe_extract(row, 'sos', 0.0),
                        'points_per_game': safe_extract(row, 'pts_per_g', 0.0),
                        'points_allowed': safe_extract(row, 'opp_pts_per_g', 0.0),
                        'pace': safe_extract(row, 'pace', 0.0),
                        'offensive_rating': safe_extract(row, 'off_rtg', 0.0),
                        'defensive_rating': safe_extract(row, 'def_rtg', 0.0),
                    }
                    
                    # Only add teams with valid data
                    if stats['wins'] > 0 or stats['losses'] > 0:
                        all_teams_stats[team_name] = stats
                        self.logger.info(f"Successfully scraped stats for {team_name}")
                    else:
                        self.logger.warning(f"Skipping {team_name} due to insufficient data")
                    
                except Exception as e:
                    self.logger.error(f"Error parsing team row: {str(e)}")
                    continue
            
            if not all_teams_stats:
                self.logger.error("No valid team statistics were found")
                return {}
                
            # Cache the results
            self._save_cache(all_teams_stats)
            
            return all_teams_stats
        
        except Exception as e:
            self.logger.error(f"Error in scrape_all_teams: {str(e)}")
            return {}

    def _save_cache(self, data: Dict[str, Any]) -> None:
        """Save scraped data to cache file."""
        try:
            data['last_updated'] = datetime.now().isoformat()
            with open(self.stats_cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")

    def load_cached_stats(self) -> Dict[str, Any]:
        """Load team statistics from cache file."""
        try:
            with open(self.stats_cache_file, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            self.logger.warning("Cache file not found. Will need to scrape fresh data.")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading cache: {str(e)}")
            return {}

def main():
    scraper = CBBStatsScraper()
    
    # Try to load cached data first
    cached_stats = scraper.load_cached_stats()
    
    # If cache is empty or older than 24 hours, scrape fresh data
    if not cached_stats or (datetime.now() - datetime.fromisoformat(cached_stats.get('last_updated', '2000-01-01'))).days >= 1:
        print("Scraping fresh team statistics...")
        stats = scraper.scrape_all_teams()
    else:
        print("Using cached team statistics...")
        stats = cached_stats
    
    # Remove the last_updated key before printing stats
    if 'last_updated' in stats:
        del stats['last_updated']
    
    # Print some sample stats
    print("\nSample Team Statistics:")
    for team_name, team_stats in list(stats.items())[:5]:  # Show first 5 teams
        if isinstance(team_stats, dict):  # Make sure team_stats is a dictionary
            print(f"\n{team_name}:")
            print(f"Win Rate: {team_stats.get('win_rate', 'N/A')}")
            print(f"Points per Game: {team_stats.get('points_per_game', 'N/A')}")
            print(f"Defensive Rating: {team_stats.get('defensive_rating', 'N/A')}")
            print(f"Strength of Schedule: {team_stats.get('sos', 'N/A')}")

if __name__ == "__main__":
    main() 